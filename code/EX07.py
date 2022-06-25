import os
import time
import gtsam
import numpy as np
import pandas as pd
from gtsam.utils import plot
import matplotlib
from matplotlib import pyplot as plt

import config as c
import utils as u
from models.camera import Camera
from models.database import DataBase
from logic.trajectory import read_ground_truth_trajectory
from logic.db_adapter import DBAdapter
from logic.PoseGraph import PoseGraph
from service.TrajectoryOptimizer2 import TrajectoryOptimizer2

# change matplotlib's backend
matplotlib.use("webagg")

##################################
#          Load Data             #
##################################

start = time.time()

K, M_left, M_right = u.read_first_camera_matrices()
Camera._K = K
Camera._RightRotation = M_right[:, :3]
Camera._RightTranslation = M_right[:, 3:]

gt_traj = read_ground_truth_trajectory()

dba = DBAdapter(data=[])
dba.tracks_db = pd.read_pickle(os.path.join(c.DATA_WRITE_PATH, 'long_tracks.pkl'))
dba.cameras_db = pd.read_pickle(os.path.join(c.DATA_WRITE_PATH, 'relative_cameras.pkl'))

# perform bundle adjustment from EX05
traj_opt = TrajectoryOptimizer2(tracks=dba.tracks_db, relative_cams=dba.cameras_db[DataBase.CAM_LEFT])
traj_opt.optimize(verbose=True)


#########

pg = PoseGraph(traj_opt.bundles)
pg.optimize_with_loops(verbose=True)

elapsed = time.time() - start

#########

num_keyframes = len(pg.keyframe_symbols)
optimized_keyframe_trajectory = np.zeros((num_keyframes, 3))
ground_truth_keyframe_trajectory = np.zeros((num_keyframes, 3))
for i, (fr_idx, symbol) in enumerate(pg.keyframe_symbols.items()):
    pose = pg._optimized_estimates.atPose3(symbol)
    cam = Camera.from_pose3(fr_idx, pose)
    optimized_keyframe_trajectory[i] = cam.calculate_coordinates()
    ground_truth_keyframe_trajectory[i] = gt_traj.T[fr_idx]

optimized_keyframe_trajectory = optimized_keyframe_trajectory.T        # shape 3xN
ground_truth_keyframe_trajectory = ground_truth_keyframe_trajectory.T  # shape 3xN
euclidean_distances = np.linalg.norm(optimized_keyframe_trajectory - ground_truth_keyframe_trajectory, ord=2, axis=0)

plt.clf()
fig, axes = plt.subplots(1, 2)
fig.set_figwidth(15)

axes[0].scatter(optimized_keyframe_trajectory[0], optimized_keyframe_trajectory[2],
                marker='o', c='r', s=5, label='estimated (GTSAM)')
axes[0].scatter(ground_truth_keyframe_trajectory[0], ground_truth_keyframe_trajectory[2],
                marker='o', c='b', s=5, label='ground truth')
axes[0].set_title("KITTI Trajectories")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Z")
axes[0].legend(loc='best')

axes[1].scatter([fr_idx for fr_idx in pg.keyframe_symbols.keys()], euclidean_distances, c='k', marker='o', s=2)
axes[1].set_title("Euclidean Distances")
axes[1].set_xlabel("Frame ID")

plt.show()




