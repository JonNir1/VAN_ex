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
        image_path = c.DATA_PATH + '\\image_0\\' + image_name
    else:
        image_path = c.DATA_PATH + '\\image_1\\' + image_name
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Load a pair of KITTY images of idx $idx into cv2.Image objects
def read_image_pair(idx: int):
    img0 = read_single_image(idx, True)
    img1 = read_single_image(idx, False)
    return img0, img1


def read_cameras():
    """
    Load camera matrices from the KITTY dataset
    Returns 3 np.arrays:
      k - Intrinsic camera matrix
      m1, m2 - Extrinsic camera matrix (left, right)
    """
    with open(os.path.join(c.DATA_PATH, 'calib.txt'), "r") as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


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


# create a cv2 feature detector
def create_detector(detector_name: str):
    if detector_name == "orb" or detector_name == "ORB":
        return cv2.ORB_create()
    if detector_name == "sift" or detector_name == "SIFT":
        return cv2.SIFT_create()
    raise NotImplementedError("We currently do not " +
                              f"support the {detector_name} detector")


# create a cv2.matcher object
def create_matcher(matcher_name: str, norm=cv2.NORM_L2, cross_check: bool = True):
    if matcher_name == "bf" or matcher_name == "BF":
        return cv2.BFMatcher(norm, cross_check=cross_check)
    if matcher_name == "flann" or matcher_name == "FLANN":
        return cv2.FlannBasedMatcher(indexParams=dict(algorithm=0, trees=5), searchParams=dict(checks=50))
    raise NotImplementedError(f"We currently do not support the \"{matcher_name}\" matcher")


def homogenize_array():
    # TODO
    pass


def dehomogenize_array():
    # TODO
    pass


def euclidean_distance():
    # TODO
    pass


def pixel_distance():
    # TODO
    pass

