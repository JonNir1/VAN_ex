from typing import Optional

import numpy as np

import config as c
import utils as u
from models.Camera2 import Camera2


class Frame2:
    """Represents a stereo rectified image-pair from the KITTI dataset"""

    MaxIndex = 3449
    MaxVerticalDistanceForInlier = 1

    def __init__(self, idx: int, left_cam: Optional[Camera2] = None):
        if idx < 0 or idx > Frame2.MaxIndex:
            raise IndexError(f"Frame index must be between 0 and {Frame2.MaxIndex}, not {idx}")
        self.id = idx
        self.left_camera = left_cam
        self.features, self.descriptors = self.__detect_and_match(idx)

    @staticmethod
    def __detect_and_match(idx: int):
        """
        Detects keypoints and extracts matches between two images of a single Frame.
        A match is only extracted if the two keypoints have a horizontal distance below a given number of pixels.

        Returns two numpy arrays:
            features - the (X_left, Y_left, X_right, Y_right) coordinates of each matched feature (rows are different features)
            descriptors - the CV2 descriptors of each (left) matched feature
        """
        img_left, img_right = u.read_image_pair(idx)
        features_list, descriptors_list = [], []

        keypoints_left, descriptors_left = c.DETECTOR.detectAndCompute(img_left, None)
        keypoints_right, descriptors_right = c.DETECTOR.detectAndCompute(img_right, None)
        all_matches = c.MATCHER.match(descriptors_left, descriptors_right)  # cv2Match
        for m in all_matches:
            left_keypoint, right_keypoint = keypoints_left[m.queryIdx], keypoints_right[m.trainIdx]
            x_l, y_l = left_keypoint.pt
            x_r, y_r = right_keypoint.pt
            if abs(y_l - y_r) > Frame2.MaxVerticalDistanceForInlier:
                continue
            features_list.append([x_l, y_l, x_r, y_r])
            descriptors_list.append(descriptors_left[m.queryIdx])

        features = np.array(features_list)
        descriptors = np.array(descriptors_list)
        return features, descriptors

    def __str__(self):
        return f"Fr{self.id}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, Frame2):
            return False
        return self.id == other.id
