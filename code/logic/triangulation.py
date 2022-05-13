import cv2
import numpy as np
from typing import List

import config as c
from models.directions import Side
from models.camera import Camera
from models.match import FrameMatch


def triangulate_matches(matches: List[FrameMatch], left_cam: Camera, right_cam: Camera) -> np.ndarray:
    """
    Reconstructs 3D points from array of paired 2D keypoints, based on the two Camera objects that created the keypoints
    @params:
        $matches: list of N keypoint-pairs to reconstruct using cv2 triangulation
        $left_cam, $right_cam: Camera objects that represent the left- and right-camera that took the stereo-corrected image

    Returns a 3D point cloud of shape (3,N) where each row is a coordinate (X, Y, Z) and each col is a point
    """
    left_pixels = np.array([(m.get_keypoint(Side.LEFT)).pt for m in matches])
    right_pixels = np.array([(m.get_keypoint(Side.RIGHT)).pt for m in matches])
    return triangulate(left_pixels.T, right_pixels.T, left_cam, right_cam)


def triangulate(pixels1: np.ndarray, pixels2: np.ndarray, cam1: Camera, cam2: Camera) -> np.ndarray:
    pixels1 = _verify_shapes(pixels1)
    pixels2 = _verify_shapes(pixels2)
    assert pixels1.shape[1] == pixels2.shape[1], "cannot triangulate unequal number of samples" + \
                                                 f"{pixels1.shape[1]} and {pixels2.shape[1]}"
    proj1 = cam1.calculate_projection_matrix()
    proj2 = cam2.calculate_projection_matrix()
    X_4d = cv2.triangulatePoints(proj1, proj2, pixels1, pixels2)
    X_4d /= (X_4d[3] + c.Epsilon)  # homogenize; add small epsilon to prevent division by 0
    return X_4d[:-1]  # return only the 3d coordinates


def _verify_shapes(pixels: np.ndarray) -> np.ndarray:
    # verified that the input is an array of shape 2xN
    assert pixels.shape[0] == 2 or pixels.shape[1] == 2, f"input must be a 2D array"
    if pixels.shape[0] != 2:
        return pixels.T
    return pixels

