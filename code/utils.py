"""
A set of utility functions to be shared between different exercises.
"""
import os.path

import cv2
import numpy as np
import config as c


# Load a single KITTY image of idx $idx into cv2.Image objects
# either load the left or right image by using the bool $is_left
def read_single_image(idx: int, is_left: bool):
    image_name = "{:06d}.png".format(idx)
    image_dir = "image_0" if is_left else "image_1"
    image_path = os.path.join(c.DATA_READ_PATH, "sequences", "00", image_dir, image_name)
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Load a pair of KITTY images of idx $idx into cv2.Image objects
def read_image_pair(idx: int):
    img0 = read_single_image(idx, True)
    img1 = read_single_image(idx, False)
    return img0, img1


def read_first_camera_matrices(path: str = ""):
    """
    Load camera matrices from the KITTY dataset
    Returns the following matrices (ndarrays):
        K - Intrinsic camera matrix
        M_left, M_right - Extrinsic camera matrix (left, right)
    """
    if path is None or path == "":
        path = os.path.join(c.DATA_READ_PATH, "sequences", "00", "calib.txt")
    with open(path, "r") as f:
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

