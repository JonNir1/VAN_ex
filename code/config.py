import os

from cv2_utils import create_detector
from models.matcher import Matcher

Epsilon = 1e-10

MAIN_DIRECTORY = "C:\\Users\\nirjo\\Documents\\University\\Masters\\Computer Vision Aided Navigation\\VAN_ex"
DATA_READ_PATH = os.path.join(MAIN_DIRECTORY, 'dataset')
DATA_WRITE_PATH = os.path.join(MAIN_DIRECTORY, "docs\\db")

DEFAULT_DETECTOR_NAME = "sift"
DETECTOR = create_detector(DEFAULT_DETECTOR_NAME)

DEFAULT_MATCHER_NAME = "flann"
SHOULD_CROSS_CHECK = False
MATCHER = Matcher(DEFAULT_MATCHER_NAME, SHOULD_CROSS_CHECK)


