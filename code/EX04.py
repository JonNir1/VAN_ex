import os
import cv2
import numpy as np
import pandas as pd

import config as c
import utils as u
from models.directions import Side
from models.frame import Frame
from models.camera import Camera
from logic.triangulation_logic import TriangulationLogic


K, M1, M2 = u.read_cameras()
cam_left = Camera(0, Side.LEFT, K, M1)
cam_right = Camera(0, Side.RIGHT, K, M2)
frame = Frame(0)

tri = TriangulationLogic(frame, cam_left, cam_right)
cloud = tri.match_and_triangulate()

print(cam_left)



