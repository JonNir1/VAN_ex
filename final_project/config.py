import os
from final_project.models.Matcher import Matcher

# MAIN_DIRECTORY = "C:\\Users\\nirjo\\Documents\\University\\Masters\\Computer Vision Aided Navigation\\VAN_ex"  # Windows path
MAIN_DIRECTORY = "/mnt/c/Users/nirjo/Documents/University/Masters/Computer Vision Aided Navigation/VAN_ex"  # WSL path
DATA_READ_PATH = os.path.join(MAIN_DIRECTORY, 'dataset')
DATA_WRITE_PATH = os.path.join(MAIN_DIRECTORY, "final_project", "outputs")


Epsilon = 1e-8
NUM_FRAMES = 3450
BUNDLE_SIZE = 10


_DEFAULT_DETECTOR_NAME = "sift"
_DEFAULT_MATCHER_NAME = "flann"
_SHOULD_CROSS_CHECK = False
_SHOULD_USE_2NN = True
DEFAULT_MATCHER = Matcher(_DEFAULT_DETECTOR_NAME, _DEFAULT_MATCHER_NAME, _SHOULD_CROSS_CHECK, _SHOULD_USE_2NN)

# Global Variable Names
FrameIdx, TrackIdx = "FrameIdx", "TrackIdx"
XL, XR, Y = "XL", "XR", "Y"
CamL, CamR = "CamL", "CamR"
Symbol = "Symbol"
InitialPose, OptPose = "InitPose", "OptPose"
AbsolutePose = "AbsPose"
