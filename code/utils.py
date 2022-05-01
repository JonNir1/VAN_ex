"""
A set of utility functions to be shared between different exercises.
"""

import os
import cv2
import numpy as np
import config as c


# Load a single KITTY image of idx $idx into cv2.Image objects
# either load the left or right image by using the bool $is_left
def read_single_image(idx: int, is_left: bool):
    image_name = "{:06d}.png".format(idx)
    if is_left:
        image_path = c.DATA_READ_PATH + '\\image_0\\' + image_name
    else:
        image_path = c.DATA_READ_PATH + '\\image_1\\' + image_name
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Load a pair of KITTY images of idx $idx into cv2.Image objects
def read_image_pair(idx: int):
    img0 = read_single_image(idx, True)
    img1 = read_single_image(idx, False)
    return img0, img1


def read_first_camera_matrices():
    """
    Load camera matrices from the KITTY dataset
    Returns the following matrices (ndarrays):
        K - Intrinsic camera matrix
        M_left, M_right - Extrinsic camera matrix (left, right)
    """
    with open(os.path.join(c.DATA_READ_PATH, 'calib.txt'), "r") as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    K = m1[:, :3]
    M_left = np.linalg.inv(K) @ m1
    M_right = np.linalg.inv(K) @ m2
    return K, M_left, M_right


def read_poses():
    """
    Load ground truth extrinsic matrices of left cameras from the KITTI trajectory.
    All cameras are calibrated w.r.t. the first left camera
    returns two lists of matrices:
        Rs are 3x3 rotation matrices
        ts are 3x1 translation vectors
    """
    Rs, ts = [], []
    file_path = os.path.join(os.getcwd(), r'dataset\poses\00.txt')
    f = open(file_path, 'r')
    for i, line in enumerate(f.readlines()):
        mat = np.array(line.split(), dtype=float).reshape((3, 4))
        Rs.append(mat[:, :3])
        ts.append(mat[:, 3:])
    return Rs, ts


def read_trajectory() -> np.ndarray:
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


def homogenize_array():
    # TODO
    pass


def dehomogenize_array():
    # TODO
    pass


def pixel_distance():
    # TODO
    pass

