import os
import cv2

DATA_PATH = os.path.join(os.getcwd(), r'dataset\sequences\00')

DEFAULT_DETECTOR = cv2. SIFT_create()
DEFAULT_MATCHER = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)