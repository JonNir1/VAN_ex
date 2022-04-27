import os

Epsilon = 1e-10

DATA_READ_PATH = os.path.join(os.getcwd(), r'dataset\sequences\00')
DATA_WRITE_PATH = os.path.join(os.getcwd(), r'docs\db')

DEFAULT_DETECTOR_NAME = "sift"
DEFAULT_MATCHER_NAME = "flann"
SHOULD_CROSS_CHECK = False


