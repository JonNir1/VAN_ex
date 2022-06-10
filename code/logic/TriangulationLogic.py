import numpy as np
import cv2
from typing import Optional

import config as c
from models.Camera2 import Camera2


def triangulate_pixels(pixels1: np.ndarray, pixels2: np.ndarray, left_cam: Optional[Camera2] = None) -> np.ndarray:
    """
    Reconstructs 3D points from arrays of 2D matched pixels, and the (left) Camera2 object that created the pixels.

    :param pixels1: a matrix of shape 2xN of the left camera's pixels
    :param pixels2: a matrix of shape 2xN of the right camera's pixels
    :param left_cam: if provided, triangulates the points relative to this camera (and the equivalent right camera).
            if not provided, triangulate based on the 0-located left camera (and the equivalent right camera).
    :return: a 3xN matrix containing (X, Y, Z) landmark coordinates based on the input pixel arrays.
    """
    pixels1 = __verify_array_shape(pixels1)
    pixels2 = __verify_array_shape(pixels2)
    assert pixels1.shape[1] == pixels2.shape[1], "cannot triangulate unequal number of samples" + \
                                                 f"{pixels1.shape[1]} and {pixels2.shape[1]}"
    if left_cam is None:
        left_cam, right_cam = Camera2.get_initial_cameras()
    else:
        left_cam, right_cam = left_cam, left_cam.calculate_right_camera()

    left_proj = left_cam.projection_matrix
    right_proj = right_cam.projection_matrix
    X_4d = cv2.triangulatePoints(left_proj, right_proj, pixels1, pixels2)
    X_4d /= (X_4d[3] + c.Epsilon)  # homogenize; add small epsilon to prevent division by 0
    return X_4d[:-1]  # return only the 3d coordinates


def __verify_array_shape(pixels: np.ndarray) -> np.ndarray:
    # verified that the input is an array of shape 2xN
    assert pixels.shape[0] == 2 or pixels.shape[1] == 2, f"input must be a 2D array"
    if pixels.shape[0] != 2:
        return pixels.T
    return pixels
