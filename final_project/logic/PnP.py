import cv2
import numpy as np
from typing import Optional, Tuple

from final_project.models.Camera import Camera

_MinSamplesNumber = 4


def pnp(points: np.ndarray, pixels: np.ndarray) -> Optional[Camera]:
    """
    Calculates a camera's [R|t] Extrinsic Matrix using cv2.solvePnP based on the 3D points and their corresponding
        2D projections onto the camera's plane
    If the PnP was successful, returns a Camera object with the resulting Extrinsic Matrix. Otherwise, returns None.
    @raises AssertionError if input dimensions are not as required for PnP
    """
    points, pixels = __verify_input(points, pixels)
    success, rotation, translation = cv2.solvePnP(objectPoints=points,         # N×3
                                                  imagePoints=pixels,          # N×2
                                                  cameraMatrix=Camera.K(),  # 3×3
                                                  distCoeffs=None, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        return None
    return Camera.from_Rt(cv2.Rodrigues(rotation)[0], translation)


def __verify_input(points: np.ndarray, pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = __verify_array_shape(points, 3, "points")
    pixels = __verify_array_shape(pixels, 2, "pixels")
    num_points = points.shape[0]
    num_pixels = pixels.shape[0]
    assert num_points == num_pixels, f"unequal number of samples for points ({num_points}) and pixels ({num_pixels})"
    assert num_points >= _MinSamplesNumber, f"cannot calculate PnP with less than {_MinSamplesNumber} samples"
    return points, pixels


def __verify_array_shape(arr: np.ndarray, dim: int, name: str) -> np.ndarray:
    assert arr.shape[0] == dim or arr.shape[1] == dim, f"Each element in array \"{name}\" should have {dim} columns"
    if arr.shape[1] == dim:
        return arr
    return arr.T


