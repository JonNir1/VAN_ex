import os
import cv2

MAIN_DIRECTORY = "C:\\Users\\nirjo\\Documents\\University\\Masters\\Computer Vision Aided Navigation\\VAN_ex"  # Windows path
# MAIN_DIRECTORY = "/mnt/c/Users/nirjo/Documents/University/Masters/Computer Vision Aided Navigation/VAN_ex"  # WSL path
DATA_READ_PATH = os.path.join(MAIN_DIRECTORY, 'dataset')
DATA_WRITE_PATH = os.path.join(MAIN_DIRECTORY, "final_project", "outputs")


Epsilon = 1e-8
NUM_FRAMES = 3450
BUNDLE_SIZE = 15

DEFAULT_DETECTOR_NAME = "sift"
DEFAULT_MATCHER_NAME = "bf"
SHOULD_CROSS_CHECK = True
SHOULD_USE_2NN = False

# Global Variable Names
FrameIdx, TrackIdx = "FrameIdx", "TrackIdx"
XL, XR, Y = "XL", "XR", "Y"
CamL, CamR = "CamL", "CamR"
Symbol = "Symbol"
InitialPose, OptPose = "InitPose", "OptPose"
AbsolutePose = "AbsPose"
FrontFrame, BackFrame = "FrontFrame", "BackFrame"
OutlierPercent = "OutlierPercent"
ErrorDiff = "ErrorDiff"


def read_images(idx: int):
    """
    Load a pair of KITTI images with the given index
    """
    image_name = "{:06d}.png".format(idx)
    left_dir = "image_0"
    left_path = os.path.join(DATA_READ_PATH, "sequences", "00", left_dir, image_name)
    left_image = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)

    right_dir = "image_1"
    right_path = os.path.join(DATA_READ_PATH, "sequences", "00", right_dir, image_name)
    right_image = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    return left_image, right_image


