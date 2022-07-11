import os
from final_project.models.Matcher import Matcher

# Windows paths:
MAIN_DIRECTORY = "C:\\Users\\nirjo\\Documents\\University\\Masters\\Computer Vision Aided Navigation\\VAN_ex"
DATA_READ_PATH = os.path.join(MAIN_DIRECTORY, 'dataset')

# WSL paths:
# MAIN_DIRECTORY = "/mnt/c/Users/nirjo/Documents/University/Masters/Computer Vision Aided Navigation/VAN_ex"
# DATA_READ_PATH = os.path.join(MAIN_DIRECTORY, 'dataset')


Epsilon = 1e-8
NUM_FRAMES = 3450


_DEFAULT_DETECTOR_NAME = "sift"
_DEFAULT_MATCHER_NAME = "flann"
_SHOULD_CROSS_CHECK = False
_SHOULD_USE_2NN = True
DEFAULT_MATCHER = Matcher(_DEFAULT_DETECTOR_NAME, _DEFAULT_MATCHER_NAME, _SHOULD_CROSS_CHECK, _SHOULD_USE_2NN)

