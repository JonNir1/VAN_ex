import numpy as np

import final_project.config as c
from final_project.models.Camera import Camera


def project(cam: Camera, landmarks: np.ndarray) -> np.ndarray:
    """
    Projects an array of N 3D points onto the Camera's plane.
    Returns an array of shape 2×N containing the projected pixels (X,Y coordinates) of each landmark
    """
    landmarks = __verify_array_shape(landmarks)
    K, R, t = cam.K, cam.R, cam.t
    projections_3d = K @ (R @ landmarks + t)  # non normalized homogeneous coordinates of shape 3xN
    hom_coordinates = projections_3d / (projections_3d[2] + c.Epsilon)  # add epsilon to avoid 0 division
    return hom_coordinates[:2]  # return only first 2 rows (x,y coordinates)


def __verify_array_shape(landmarks: np.ndarray) -> np.ndarray:
    assert landmarks.shape[0] == 3 or landmarks.shape[1] == 3, f"landmarks should be 3D points"
    if landmarks.shape[0] == 3:
        return landmarks
    return landmarks.T

