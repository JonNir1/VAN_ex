import os
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

#################################
#        Question 6.1           #
#      extract covariance       #
#################################

bundle0 = traj_opt.bundles[0]
marginals0 = gtsam.Marginals(bundle0.graph, bundle0.optimized_estimates)

start_frame_idx = min(bundle0.frame_symbols.keys())
start_frame_symbol = bundle0.frame_symbols[start_frame_idx]
end_frame_idx = max(bundle0.frame_symbols.keys())
end_frame_symbol = bundle0.frame_symbols[end_frame_idx]
end_frame_pose = bundle0.optimized_estimates.atPose3(end_frame_symbol)

keys = gtsam.KeyVector()
keys.append(start_frame_symbol)
keys.append(end_frame_symbol)
marginal_cov = marginals0.jointMarginalCovariance(keys).fullMatrix()
info_mat0 = np.linalg.inv(marginal_cov)
relative_cov0 = np.linalg.inv(info_mat0[-6:, -6:])

print(end_frame_pose)
print(relative_cov0)

plt.clf()
plot.plot_trajectory(fignum=0, values=bundle0.optimized_estimates, scale=1,
                     title="First Bundle Trajectory", marginals=marginals0)
plt.show()


#################################
#         Question 6.2          #
#       Build Pose Graph        #
#################################

pg = PoseGraph(traj_opt.bundles)
pre_err = pg.error
print(f"Pre-Optimization Error:\t{pre_err:.5f}")
pg.optimize()
post_err = pg.error
print(f"Post-Optimization Error:\t{post_err:.5f}")

# plot keyframe locations provided as initial estimates
plt.clf()
plot.plot_trajectory(fignum=1, values=pg._initial_estimates, scale=1, title="Pose Graph :: Initial Estimates")
plt.show()

# plot keyframe locations after optimization
plt.clf()
plot.plot_trajectory(fignum=2, values=pg._optimized_estimates, scale=1, title="Pose Graph :: Optimization Results")
plt.show()

# plot keyframe locations after optimization - with covariances
marginals = gtsam.Marginals(pg._factor_graph, pg._optimized_estimates)
plt.clf()
plot.plot_trajectory(fignum=3, values=pg._optimized_estimates, scale=1, marginals=marginals,
                     title="Pose Graph :: Optimization Results with Covariances")
plt.show()

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


