import numpy as np
import cv2

import config as c
import utils as u
from models.Camera2 import Camera2
from models.Frame2 import Frame2
from models.frame import Frame
from logic.TriangulationLogic import triangulate_pixels
from logic.PnPLogic import pnp
from logic.RANSACLogic import RANSAC

from models.directions import Side, Position
from models.camera import Camera
from logic.pnp import compute_front_cameras
from logic.triangulation import triangulate_matches, triangulate
f0 = Frame(0)
f1 = Frame(1)
K, M_left, M_right = u.read_first_camera_matrices()
f0.left_camera = Camera(0, Side.LEFT, M_left)
f0.right_camera = Camera(0, Side.RIGHT, M_right)
mutual_matches = c.MATCHER.match_between_frames(f0, f1)
c1_l, c1_r = compute_front_cameras(mutual_matches, f0.left_camera, f0.right_camera)

back_landmarks_3d = triangulate_matches([m.get_frame_match(Position.BACK) for m in mutual_matches],
                                        f0.left_camera, f0.right_camera)
front_left_pixels = np.array([(m.get_keypoint(Side.LEFT, Position.FRONT)).pt for m in mutual_matches])


fr0 = Frame2(0)


fr1 = Frame2(1)
fr2 = Frame2(2)

left_cam0, right_cam0 = Camera2.get_initial_cameras()
K = left_cam0.intrinsic_matrix


m01 = c.MATCHER.match(fr0.descriptors, fr1.descriptors)
back_features = np.array([fr0.features[m.queryIdx] for m in m01])
front_features = np.array([fr0.features[m.trainIdx] for m in m01])

back_left_pixels, back_right_right = back_features.T[:2], back_features.T[2:]
front_left_pixels2 = front_features.T[:2]

landmark_3d_coordinates = triangulate_pixels(back_left_pixels, back_right_right)


fl_cam1 = pnp(back_features, front_features)

fl_cam1_ransac, supporter_idxs = RANSAC().run(back_features, front_features, True)


