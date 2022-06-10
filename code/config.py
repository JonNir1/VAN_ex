import os
import cv2

from models.matcher import Matcher

Epsilon = 1e-8

# Windows paths:
# MAIN_DIRECTORY = "C:\\Users\\nirjo\\Documents\\University\\Masters\\Computer Vision Aided Navigation\\VAN_ex"
# DATA_READ_PATH = os.path.join(MAIN_DIRECTORY, 'dataset')
# DATA_WRITE_PATH = os.path.join(MAIN_DIRECTORY, "docs\\db")

# WSL paths:
MAIN_DIRECTORY = "/mnt/c/Users/nirjo/Documents/University/Masters/Computer Vision Aided Navigation/VAN_ex"
DATA_READ_PATH = os.path.join(MAIN_DIRECTORY, 'dataset')
DATA_WRITE_PATH = os.path.join(MAIN_DIRECTORY, "docs/db")


def create_detector(detector_name: str):
    # create a cv2 feature detector
    detector_name = detector_name.upper()
    if detector_name == "ORB":
        return cv2.ORB_create()
    if detector_name == "SIFT":
        return cv2.SIFT_create()
    raise NotImplementedError(f"We currently do not support the {detector_name} detector")


DEFAULT_DETECTOR_NAME = "sift"
DETECTOR = create_detector(DEFAULT_DETECTOR_NAME)

DEFAULT_MATCHER_NAME = "flann"
SHOULD_CROSS_CHECK = False
MATCHER = Matcher(DEFAULT_MATCHER_NAME, SHOULD_CROSS_CHECK)


