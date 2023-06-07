import os
import subprocess
import numpy as np
import cv2
from collections import deque
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.video.source import get_video_frames_generator
from supervision.video.dataclasses import VideoInfo
import collections
from typing import List
from supervision.draw.color import ColorPalette
import scipy
from scipy.optimize import linear_sum_assignment


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version)."""
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements."""
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.7
    track_buffer: int = 70
    match_thresh: float = 0.9
    aspect_ratio_thresh: float = 3.5
    min_box_area: float = 2
    mot20: bool = False


def non_max_suppression_fast(boxes, scores, class_ids, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], [], []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("float"), scores[pick], class_ids[pick]


def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


def match_detections_with_tracks(detections: Detections, tracks: List[STrack]) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return Detections(
            xyxy=np.empty((0, 4)),
            confidence=np.empty((0,)),
            class_id=np.empty((0,), dtype=int),
            tracker_id=np.empty((0,), dtype=int)
        )

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return Detections(
        xyxy=detections.xyxy,
        confidence=detections.confidence,
        class_id=detections.class_id,
        tracker_id=np.array(tracker_ids)
    )


YOLO_MODEL_PATH = "yolov8m.pt"
SOURCE_VIDEO_PATH = r"E:\truckk\asd.mp4"
TARGET_VIDEO_PATH = r"E:\truckk\truck-result.mp4"
CLASS_NAMES_DICT = {7: "truck", }
color_palette = ColorPalette()
confidence_threshold = 0.7
capture_duration = 7  # in seconds

converted_video_path = os.path.join(os.path.dirname(SOURCE_VIDEO_PATH), "converted_video.mp4")
ffmpeg_command = f"ffmpeg -i {SOURCE_VIDEO_PATH} -c:v libx264 -crf 23 -c:a copy {converted_video_path}"
subprocess.call(ffmpeg_command, shell=True)

cap = cv2.VideoCapture(converted_video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
prev_frame_buffer = deque(maxlen=int(frame_rate * 14) + 1)

byte_tracker = BYTETracker(BYTETrackerArgs())
generator = get_video_frames_generator(converted_video_path)
box_annotator = BoxAnnotator(color=color_palette, thickness=4, text_thickness=4, text_scale=2)

truck_count = 0
truck_detected = False

model = YOLO(YOLO_MODEL_PATH)
model.fuse()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
full_video_writer = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, frame_rate, (frame_width, frame_height))
active_trucks = {}
inactive_trucks = {}
current_truck_id = None
current_truck_buffer = None
frame_counter = 0
capture_active = False

kalman_filter = KalmanFilter()
truck_count = 0
counted_trucks = set()
# Set a grace period of, say, 30 frames.
grace_period = 30

# This dictionary will map tracker IDs to the number of grace frames left for them.
recently_disappeared = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    prev_frame_buffer.append(frame)

    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    boxes, scores, class_ids = non_max_suppression_fast(boxes, scores, class_ids, 0.3)

    if len(boxes) == 0:
        continue

    detections = Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=class_ids
    )

    mask = np.logical_and(detections.class_id == 7, detections.confidence > confidence_threshold)
    detections.filter(mask=mask, inplace=True)

    if len(detections) > 0:
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape,
            filter_class=7
        )

        corrected_detections = match_detections_with_tracks(detections, tracks)

        ids_in_current_frame = set()

        for detection in corrected_detections:
            tracker_id = detection[3]
            ids_in_current_frame.add(tracker_id)

            if tracker_id is not None and tracker_id not in active_trucks and tracker_id not in recently_disappeared:
                truck_count += 1
                active_trucks[tracker_id] = {
                    'frame_count': 0,
                    'truck_count': truck_count,
                    'video_writer': cv2.VideoWriter(f"E:/truckk/truck_{truck_count}_result.mp4", fourcc, frame_rate,
                                                    (frame_width, frame_height)),
                    'buffer': collections.deque(maxlen=int(frame_rate * 14) + 1),
                }

            if tracker_id in active_trucks:
                current_truck_id = tracker_id
                current_truck_buffer = active_trucks[tracker_id]['buffer']

                truck = active_trucks[tracker_id]
                truck['frame_count'] += 1
                truck['buffer'].extend(prev_frame_buffer)
                truck['buffer'].append(frame)

                if truck['frame_count'] > frame_rate * 14:
                    truck['video_writer'].write(truck['buffer'].popleft())

        for tracker_id in list(active_trucks.keys()):
            if tracker_id not in ids_in_current_frame:
                recently_disappeared[tracker_id] = active_trucks.pop(tracker_id)
                recently_disappeared[tracker_id]['grace_frames_left'] = grace_period

                # Write remaining frames to the video file.
                truck = recently_disappeared[tracker_id]
                while truck['buffer']:
                    truck['video_writer'].write(truck['buffer'].popleft())
                if len(truck['buffer']) == 0:
                    truck['video_writer'].release()

        for tracker_id in list(recently_disappeared.keys()):
            if tracker_id in ids_in_current_frame:
                # Reopen the video writer for this truck.
                recently_disappeared[tracker_id]['video_writer'] = cv2.VideoWriter(
                    f"E:/truckk/truck_{recently_disappeared[tracker_id]['truck_count']}_result.mp4", fourcc, frame_rate,
                    (frame_width, frame_height))
                active_trucks[tracker_id] = recently_disappeared.pop(tracker_id)
            else:
                recently_disappeared[tracker_id]['grace_frames_left'] -= 1
                if recently_disappeared[tracker_id]['grace_frames_left'] <= 0:
                    del recently_disappeared[tracker_id]

        for tracker_id, truck in list(inactive_trucks.items()):
            while truck['buffer']:
                truck['video_writer'].write(truck['buffer'].popleft())
            if len(truck['buffer']) == 0:
                truck['video_writer'].release()
                del inactive_trucks[tracker_id]

        labels = [
            f"# ({truck_count} trucks)" if class_id == 7 else f"#{tracker_id} {CLASS_NAMES_DICT[class_id]}"
            for _, confidence, class_id, tracker_id
            in corrected_detections
        ]

        frame = box_annotator.annotate(frame, detections=corrected_detections, labels=labels)

        frame_counter = 0
        capture_active = True

    full_video_writer.write(frame)

    if capture_active:
        frame_counter += 1

        if frame_counter <= frame_rate * capture_duration:
            if current_truck_buffer is not None:
                current_truck_buffer.append(frame)
        else:
            capture_active = False

full_video_writer.release()

for truck in list(active_trucks.values()):
    while truck['buffer']:
        truck['video_writer'].write(truck['buffer'].popleft())
    truck['video_writer'].release()

for truck in list(inactive_trucks.values()):
    while truck['buffer']:
        truck['video_writer'].write(truck['buffer'].popleft())
    truck['video_writer'].release()

cap.release()
cv2.destroyAllWindows()

