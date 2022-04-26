import os
import cv2
import numpy as np
import pandas as pd
import random

from typing import Optional

import config as c
import utils as u

from models.directions import Side, Position
from models.camera import Camera
from models.frame import Frame
from models.match import FrameMatch, MutualMatch
from logic.keypoints_matching import detect_and_match, match_between_frames
from logic.triangulation import triangulate
from logic.pnp import compute_front_cameras
from logic.ransac import Ransac



left_cam0, right_cam0 = Camera.read_first_cameras()

frame0 = Frame(0)
frame1 = Frame(1)

cons, non_cons = match_between_frames(frame0, frame1)

ransac = Ransac()
l_cam1, r_cam1, supporters = ransac.run(cons, left_cam0, right_cam0, True)



