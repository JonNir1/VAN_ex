import os
import time
import cv2
import gtsam
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from typing import List, Optional

import final_project.config as c
import final_project.camera_utils as cu
from final_project.models.DataBase import DataBase
from final_project.models.Camera import Camera
from final_project.logic.Bundle import Bundle
from final_project.service.InitialEstimateCalculator import IECalc

# change matplotlib's backend
matplotlib.use("webagg")

###################################################


def init(N: int = 100):
    mtchr = c.DEFAULT_MATCHER
    iec = IECalc(matcher=mtchr)
    frames = iec.process(num_frames=N, verbose=True)
    database = DataBase(frames).prune_short_tracks(3)

    cameras_df = database._cameras_db.to_frame()
    cameras_df[c.Symbol] = database._cameras_db.index.map(lambda idx: gtsam.symbol('c', idx))
    abs_cameras = cu.convert_to_absolute_cameras(database._cameras_db)
    abs_poses = [cu.calculate_gtsam_pose(abs_cam) for abs_cam in abs_cameras]
    cameras_df[c.AbsolutePose] = abs_poses

    tracks_df = database._tracks_db
    tracks_df[c.Symbol] = tracks_df.index.get_level_values(c.TrackIdx).map(lambda idx: gtsam.symbol('l', idx)).astype(int)
    return tracks_df, cameras_df


tracks_df, cameras_df = init()

###############

b0_frame_idxs = np.arange(10)
b0_cams = cameras_df[cameras_df.index.get_level_values(level=c.FrameIdx).isin(b0_frame_idxs)]
b0_tracks = tracks_df[tracks_df.index.get_level_values(level=c.FrameIdx).isin(b0_frame_idxs)]
b0 = Bundle(b0_tracks, b0_cams)
b0.adjust()
print("0")

b1_frame_idxs = np.arange(10)
b1_cams = cameras_df[cameras_df.index.get_level_values(level=c.FrameIdx).isin(b1_frame_idxs)]
b1_tracks = tracks_df[tracks_df.index.get_level_values(level=c.FrameIdx).isin(b1_frame_idxs)]
b1 = Bundle(b1_tracks, b1_cams)
b1.adjust()
print("1")

# est_traj = Trajectory.from_relative_cameras(database.cameras)
# gt_traj = Trajectory.read_ground_truth(num_frames=N)
# dist = est_traj.calculate_distance(gt_traj)

# fig, axes = plt.subplots(1, 2)
# fig.suptitle('KITTI Trajectories')
# axes[0].scatter(est_traj.X, est_traj.Z, marker="o", c='b', label="PnP")
# axes[0].scatter(gt_traj.X, gt_traj.Z, marker="x", c='k', label="GT")
# axes[0].set_title("Trajectories")
# axes[0].set_xlabel("$m$")
# axes[0].set_ylabel("$m$")
# axes[0].legend(loc='best')
#
# axes[1].scatter([i for i in range(N)], dist, c='k', marker='*', s=1)
# axes[1].set_title("Euclidean Distance")
# axes[1].set_xlabel("Frame")
# axes[1].set_ylabel("$m$")
# fig.set_figwidth(10)
# plt.show()



