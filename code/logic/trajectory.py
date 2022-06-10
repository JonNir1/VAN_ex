import os
import numpy as np
from typing import List

import config as c
from models.frame import Frame


def calculate_trajectory(frames: List[Frame]) -> np.ndarray:
    """
    Calculates the left-camera trajectory of all frames.
    Returns a 3xN array representing the 3D location of all (left-)cameras of the provided Frames
    """
    num_samples = len(frames)
    trajectory = np.zeros((num_samples, 3))
    for i, fr in enumerate(frames):
        l_cam = fr.left_camera
        if l_cam is None:
            continue
        trajectory[i] = l_cam.calculate_coordinates()
    return trajectory.T


def read_ground_truth_trajectory() -> np.ndarray:
    """
    Load ground truth extrinsic matrices of left cameras from the KITTI dataset,
        and use them to calculate the camera positions in 3D coordinates.
    Returns a 3xN array representing the position of each (left-)camera
    """
    Rs, ts = read_poses()
    num_samples = len(Rs)
    trajectory = np.zeros((num_samples, 3))
    for i in range(num_samples):
        R, t = Rs[i], ts[i]
        trajectory[i] -= (R.T @ t).reshape((3,))
    return trajectory.T


def read_poses():
    """
    Load ground truth extrinsic matrices of left cameras from the KITTI trajectory.
    All cameras are calibrated w.r.t. the first left camera
    returns two lists of matrices:
        Rs are 3x3 rotation matrices
        ts are 3x1 translation vectors
    """
    Rs, ts = [], []
    trajectory_path = os.path.join(c.DATA_READ_PATH, 'poses', '00.txt')
    f = open(trajectory_path, 'r')
    for i, line in enumerate(f.readlines()):
        mat = np.array(line.split(), dtype=float).reshape((3, 4))
        Rs.append(mat[:, :3])
        ts.append(mat[:, 3:])
    return Rs, ts

