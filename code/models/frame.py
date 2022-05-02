import cv2
import numpy as np
from typing import Optional

import config as c
import utils as u
from models.match import FrameMatch
from models.camera import Camera


class Frame:
    """
    Represents a stereo rectified image-pair from the KITTI dataset
    """

    MaxIndex = 3449
    MaxVerticalDistanceForInlier = 1

    def __init__(self, idx: int, left_cam: Optional[Camera] = None, right_cam: Optional[Camera] = None):
        if idx < 0:
            raise IndexError(f"Frame index must be between 0 and {self.MaxIndex}, not {idx}")
        self.id = idx
        self.left_camera = left_cam
        self.right_camera = right_cam
        self.inlier_matches = list[FrameMatch]()
        self.inlier_descriptors = []
        self.next_frame_tracks_count = 0
        self.match_to_track_id = dict[tuple[cv2.KeyPoint, cv2.KeyPoint], int]()
        self._detect_and_match()

    def find_tracks(self, left_kp: cv2.KeyPoint) -> list[int]:
        # returns all tracks that include $left_kp
        all_tracks = []
        inliers = [fm for fm in self.inlier_matches if fm.left_keypoint == left_kp]
        for inlr in inliers:
            track_id = self.match_to_track_id.get(inlr)
            if track_id is not None:
                all_tracks.append(track_id)
        return all_tracks

    def _detect_and_match(self):
        """
        Detects keypoints and extracts matches between two images of a single Frame.
        A match is only extracted if the two keypoints have a horizontal distance below a given number of pixels.
        """
        img_left, img_right = u.read_image_pair(self.id)
        keypoints_left, descriptors_left = c.DETECTOR.detectAndCompute(img_left, None)
        keypoints_right, descriptors_right = c.DETECTOR.detectAndCompute(img_right, None)
        all_matches = c.MATCHER.match(descriptors_left, descriptors_right)
        left_descriptors = []
        for m in all_matches:
            left_kp = keypoints_left[m.queryIdx]
            right_kp = keypoints_right[m.trainIdx]
            left_descriptor = descriptors_left[m.queryIdx]
            vertical_dist = abs(left_kp.pt[1] - right_kp.pt[1])
            if vertical_dist <= self.MaxVerticalDistanceForInlier:
                self.inlier_matches.append(FrameMatch(left_kp, right_kp))
                left_descriptors.append(left_descriptor)  # store for between-frame matches
        self.inlier_descriptors = np.array(left_descriptors)

    def __str__(self):
        return f"Frame{self.id}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Frame):
            return False
        get_id_attr = getattr(other, 'get_id', None)
        if get_id_attr is None:
            return False
        if not callable(get_id_attr):
            return False
        return self.id == other.id
