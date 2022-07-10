import cv2
import numpy as np
from typing import Tuple, Optional

import final_project.config as c
from final_project.models.Camera import Camera


def triangulate(pixels1: np.ndarray, pixels2: np.ndarray,
                left_cam: Camera, right_cam: Optional[Camera] = None) -> np.ndarray:
    """
    Reconstructs 3D points from arrays of 2D matched pixels, and the (left) Camera object that created the pixels.
    :param pixels1: a matrix of shape 2×N of the left camera's pixels
    :param pixels2: a matrix of shape 2×N of the right camera's pixels
    :param left_cam: triangulates $pixels1 relative to this camera
    :param right_cam: if provided, triangulates $pixels2 relative to this camera. If not, it is inferred from $left_cam

    :raises: AssertionError if the pixel arrays are not 2D with same amount of samples (2×N)
    :return: a 3xN matrix containing (X, Y, Z) landmark coordinates based on the input pixel arrays.
    """
    pixels1, pixels2 = __verify_input(pixels1, pixels2)
    right_cam = right_cam if right_cam is not None else left_cam.get_right_camera()
    left_proj_mat = left_cam.calculate_projection_matrix()
    right_proj_mat = right_cam.calculate_projection_matrix()
    X_4d = cv2.triangulatePoints(left_proj_mat, right_proj_mat, pixels1, pixels2)
    X_4d /= (X_4d[3] + c.Epsilon)  # homogenize; add small epsilon to prevent division by 0
    return X_4d[:-1]  # return only the 3d coordinates


def __verify_input(pixels1: np.ndarray, pixels2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pixels1 = __verify_array_shape(pixels1)
    pixels2 = __verify_array_shape(pixels2)
    assert pixels1.shape[1] == pixels2.shape[
        1], f"cannot triangulate unequal number of samples: {pixels1.shape[1]} and {pixels2.shape[1]}"
    return pixels1, pixels2


def __verify_array_shape(pixels: np.ndarray) -> np.ndarray:
    # verified that the input is an array of shape 2×N
    assert pixels.shape[0] == 2 or pixels.shape[1] == 2, f"input must be a 2D array"
    if pixels.shape[0] == 2:
        return pixels
    return pixels.T
