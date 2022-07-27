import os
import cv2

from models.matcher import Matcher, create_detector

Epsilon = 1e-8

# Windows paths:
# MAIN_DIRECTORY = "C:\\Users\\nirjo\\Documents\\University\\Masters\\Computer Vision Aided Navigation\\VAN_ex"
# DATA_READ_PATH = os.path.join(MAIN_DIRECTORY, 'dataset')
# DATA_WRITE_PATH = os.path.join(MAIN_DIRECTORY, "docs\\db")

# WSL paths:
MAIN_DIRECTORY = "/mnt/c/Users/nirjo/Documents/University/Masters/Computer Vision Aided Navigation/VAN_ex"
DATA_READ_PATH = os.path.join(MAIN_DIRECTORY, 'dataset')
DATA_WRITE_PATH = os.path.join(MAIN_DIRECTORY, "docs/db")

DEFAULT_DETECTOR_NAME = "sift"
DEFAULT_MATCHER_NAME = "BF"
SHOULD_CROSS_CHECK = True
SHOULD_USE_2NN = False

DETECTOR = create_detector(DEFAULT_DETECTOR_NAME)
MATCHER = Matcher(DEFAULT_MATCHER_NAME, SHOULD_CROSS_CHECK, SHOULD_USE_2NN)


