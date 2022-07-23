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
from final_project.models.Trajectory import Trajectory
from final_project.logic.Bundle import Bundle
from final_project.service.InitialEstimateCalculator import IECalc
from final_project.service.BundleAdjustment import BundleAdjustment

# change matplotlib's backend
matplotlib.use("webagg")

###################################################


def init(N: int = 3450):
    mtchr = c.DEFAULT_MATCHER
    iec = IECalc(matcher=mtchr)
    frames = iec.process(num_frames=N, verbose=True)
    database = DataBase(frames).prune_short_tracks(3)
    database.to_pickle()
    return database


# db = init()
db = DataBase.from_pickle("tracksdb_23072022_1751", "camerasdb_23072022_1751")
ba = BundleAdjustment(db._tracks_db, db._cameras_db)
ba.optimize(verbose=True)
ba_cameras = ba.extract_cameras()

###############

pnp_traj = Trajectory.from_relative_cameras(db._cameras_db)
ba_traj = Trajectory.from_relative_cameras(ba_cameras)
gt_traj = Trajectory.read_ground_truth()

pnp_dist = pnp_traj.calculate_distance(gt_traj)
ba_dist = ba_traj.calculate_distance(gt_traj)

fig, axes = plt.subplots(1, 2)
fig.suptitle('KITTI Trajectories')
axes[0].scatter(pnp_traj.X, pnp_traj.Z, marker="o", c='b', s=2, label="PnP")
axes[0].scatter(ba_traj.X, ba_traj.Z, marker="^", c='g', s=2, label="BA")
axes[0].scatter(gt_traj.X, gt_traj.Z, marker="x", c='k', s=2, label="GT")
axes[0].set_title("Trajectories")
axes[0].set_xlabel("$m$")
axes[0].set_ylabel("$m$")
axes[0].legend(loc='best')

axes[1].scatter([i for i in range(c.NUM_FRAMES)], pnp_dist, c='b', marker='o', s=1, label="PnP")
axes[1].scatter([i for i in range(c.NUM_FRAMES)], ba_dist, c='g', marker='^', s=1, label="BA")
axes[1].set_title("Euclidean Distance")
axes[1].set_xlabel("Frame")
axes[1].set_ylabel("$m$")
axes[1].legend(loc='best')
fig.set_figwidth(12)
plt.show()





