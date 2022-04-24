import cv2
import numpy as np

import config as c
from models.frame import Frame
from models.camera import Camera


class TriangulationLogic:

    def __init__(self, frame: Frame, left_cam: Camera, right_cam: Camera):
        self._frame = frame
        self._left_camera = left_cam
        self._right_camera = right_cam

    def get_idx(self):
        return self._frame.idx

    def match_and_triangulate(self) -> np.ndarray:
        """
        For the current Frame ($self._frame), returns a 3xN array representing a cloud of 3D points.
        Each point corresponds to a keypoint match between the two images
        """
        proj_left = self._left_camera.calculate_projection_matrix()
        proj_right = self._right_camera.calculate_projection_matrix()
        left_pixels, right_pixels = self._extract_keypoint_coordinates()

        X_4d = cv2.triangulatePoints(proj_left, proj_right, left_pixels.T, right_pixels.T)
        X_4d /= (X_4d[3] + c.Epsilon)  # homogenize; add small epsilon to prevent division by 0

        # return only the 3d coordinates
        return X_4d[:-1]

    def _extract_keypoint_coordinates(self) -> (np.ndarray, np.ndarray):
        kps_left, _, kps_right, _, matches = self._frame.detect_and_match()
        left_coordinates, right_coordinates = [], []
        for m in matches:
            left_kp_idx, right_kp_idx = m.queryIdx, m.trainIdx
            left_pixel = kps_left[left_kp_idx].pt
            right_pixel = kps_right[right_kp_idx].pt
            left_coordinates.append(left_pixel)
            right_coordinates.append(right_pixel)
        return np.array(left_coordinates), np.array(right_coordinates)


