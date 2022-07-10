import os
import numpy as np
import cv2

import final_project.config as c


def read_images(idx: int):
    """
    Load a pair of KITTI images with the given index
    """
    image_name = "{:06d}.png".format(idx)
    left_dir = "image_0"
    left_path = os.path.join(c.DATA_READ_PATH, "sequences", "00", left_dir, image_name)
    left_image = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)

    right_dir = "image_1"
    right_path = os.path.join(c.DATA_READ_PATH, "sequences", "00", right_dir, image_name)
    right_image = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    return left_image, right_image


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
