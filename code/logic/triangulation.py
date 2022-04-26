import cv2
import numpy as np

import config as c
from models.directions import Side
from models.camera import Camera
from models.match import FrameMatch


def triangulate(matches: list[FrameMatch], left_cam: Camera, right_cam: Camera) -> np.ndarray:
    """
    Reconstructs 3D points from array of paired 2D keypoints, based on the two Camera objects that created the keypoints
    @params:
        $matches: list of N keypoint-pairs to reconstruct using cv2 triangulation
        $left_cam, $right_cam: Camera objects that represent the left- and right-camera that took the stereo-corrected image

    Returns a 3D point cloud of shape (3,N) where each row is a coordinate (X, Y, Z) and each col is a point
    """
    proj_left = left_cam.calculate_projection_matrix()
    proj_right = right_cam.calculate_projection_matrix()
    left_pixels = np.array([(m.get_keypoint(Side.LEFT)).pt for m in matches])
    right_pixels = np.array([(m.get_keypoint(Side.RIGHT)).pt for m in matches])

    X_4d = cv2.triangulatePoints(proj_left, proj_right, left_pixels.T, right_pixels.T)
    X_4d /= (X_4d[3] + c.Epsilon)  # homogenize; add small epsilon to prevent division by 0
    return X_4d[:-1]  # return only the 3d coordinates


