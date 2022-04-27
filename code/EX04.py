import os
import cv2
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

from typing import Optional

import config as c
import utils as u
from models.directions import Side, Position
from models.camera import Camera
from models.frame import Frame
from models.track import Track
from models.match import FrameMatch, MutualMatch
from logic.keypoints_matching import detect_and_match, match_between_frames
from logic.triangulation import triangulate
from logic.pnp import compute_front_cameras
from logic.ransac import Ransac

from service.trajectory_processor import estimate_trajectory
from models.db_adapter import DBAdapter


# real_traj = u.read_trajectory()
# real_traj_500 = real_traj[:, :500]


all_frames, est_traj = estimate_trajectory(50, verbose=True)
dba = DBAdapter(all_frames)
db = dba.build_database()




# fig, ax = plt.subplots()
# fig.suptitle('KITTI Trajectories')
# n = est_traj.shape[1]
# markers_sizes = np.ones((n,))
# markers_sizes[[i for i in range(n) if i % 50 == 0]] = 15
# markers_sizes[0], markers_sizes[-1] = 50, 50
# ax.scatter(est_traj[0], est_traj[2], marker="o", s=markers_sizes, c=est_traj[1], cmap="gray", label="estimated")
# ax.scatter(real_traj_500[0], real_traj_500[2], marker="x", s=markers_sizes, c=real_traj_500[1], label="ground truth")
# ax.set_title("Trajectories")
# ax.legend(loc='best')
# fig.set_figwidth(10)
# plt.show()






mini_dfs = {}
cam_dfs = {}
for i, fr in enumerate(all_frames):
    if i > 5:
        break
    df = pd.DataFrame({tr.get_id() : (kp_l.pt[0], kp_r.pt[0], kp_l.pt[1])
                       for tr, (kp_l, kp_r) in fr.get_tracks().items()}).T
    df.rename(columns={0: "X_L", 1: "X_R", 2: "Y"}, inplace=True)
    df.index.name = "TrackIdx"
    mini_dfs[fr.get_id()] = df
    cam_dfs[fr.get_id()] = [fr.left_camera, fr.right_camera]

df = pd.concat(mini_dfs)
df.index.set_names(["fr_idx", "tr_idx"], inplace=True)

df['cam_l'] = [f.left_camera.extrinsic_matrix for i, f in enumerate(all_frames) if i <= 5 for j in range(len(f.get_tracks()))]


