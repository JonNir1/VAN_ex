import numpy as np
import cv2
from typing import Optional, Tuple

from models.Camera2 import Camera2
from logic.TriangulationLogic import triangulate_pixels

MinPointsCount = 4


def pnp(back_features: np.ndarray, front_features: np.ndarray,
        back_left_cam: Optional[Camera2] = None) -> Optional[Camera2]:
    """
    Calculate the front-left Camera2 object, based on PnP computation of the 3D points extracted from the back cameras,
        alongside the 2D points extracted from the front-left camera.
    Uses cv2.solvePnP to compute the front-left Camera.

    :raises: AssertionError if there are less than 4 features to base PnP on.
    :returns: a Cameras2 if the calculation was successful, or None if PnP was unsuccessful
    """
    back_features = __verify_shape(back_features)
    front_features = __verify_shape(front_features)
    assert back_features.shape[1] == front_features.shape[1], f"Back & Front Features count mismatch"

    # triangulate features relative to back camera
    back_left_cam, back_right_cam = __calculate_back_cameras(back_left_cam)
    back_left_pixels, back_right_right = back_features[:2], back_features[2:]
    landmark_3d_coordinates = triangulate_pixels(back_left_pixels, back_right_right, back_left_cam)  # shape (3, N)

    # use cv2 solvePnP:
    front_left_pixels = front_features[:2]  # shape (2, N)
    K = back_left_cam.intrinsic_matrix
    success, rotation, translation = cv2.solvePnP(objectPoints=np.expand_dims(landmark_3d_coordinates.T, 0),  # shape (1, N, 3)
                                                  imagePoints=np.expand_dims(front_left_pixels.T, 0),         # shape (1, N, 2)
                                                  cameraMatrix=K, distCoeffs=None, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        return None

    front_ext = __arrange_external_matrix(rotation, translation)
    return Camera2(front_ext)


def __verify_shape(features: np.ndarray) -> np.ndarray:
    # make sure features are arranged as a 4xN matrix (Xl, Yl, Xr, Yr) as rows
    assert features.shape[0] == 4 or features.shape[1] == 4, f"Features matrix must contains four columns"
    if features.shape[0] == 4:
        return features
    return features.T


def __calculate_back_cameras(back_left_cam: Optional[Camera2] = None) -> Tuple[Camera2, Camera2]:
    """
    Returns a pair of stereo-rectified Camera2 objects.
        If a $back_left_cam is provided, a right camera is calculated to match it and this pair is returned
        Otherwise, returns a 0-located left camera and the equivalent right camera.
    """
    if back_left_cam is None:
        back_left_cam, back_right_cam = Camera2.get_initial_cameras()
    else:
        back_left_cam, back_right_cam = back_left_cam, back_left_cam.calculate_right_camera()
    return back_left_cam, back_right_cam


def __arrange_external_matrix(rotation, translation) -> np.ndarray:
    R = cv2.Rodrigues(rotation)[0]
    t = translation if translation.shape == (3, 1) else translation.reshape((3,1))
    return np.hstack([R, t])
