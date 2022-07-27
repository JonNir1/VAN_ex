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
traj_cameras = pd.Series(traj_opt.extract_all_relative_cameras())
# traj_cameras.to_pickle(os.path.join(c.DATA_WRITE_PATH, "bundle_cameras"))


#########

pg = PoseGraph(traj_opt.bundles)
pg.optimize_with_loops(verbose=True)
pg_cameras = pg.extract_relative_cameras()
# pg_cameras.to_pickle(os.path.join(c.DATA_WRITE_PATH, "posegraph_cameras"))

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


##############################
#       Question 5A          #
#    Draw Loop Matches       #
##############################

from models.frame import Frame
from logic.ransac import Ransac

early_frame_idx, late_frame_idx = 84, 1526

default_cam_left = Camera.create_default()
default_cam_left.idx = early_frame_idx
default_cam_right = default_cam_left.calculate_right_camera()

early_frame = Frame(early_frame_idx, left_cam=default_cam_left, right_cam=default_cam_right)
late_frame = Frame(late_frame_idx)
matches = c.MATCHER.match_between_frames(early_frame, late_frame)
fl_cam, fr_cam, supporters = Ransac().run(matches, bl_cam=early_frame.left_camera, br_cam=early_frame.right_camera)
fl_cam.idx = late_frame_idx
fr_cam.idx = late_frame_idx
outliers = [m for m in matches if m not in supporters]

plt.clf()
fig, axes = plt.subplots(1, 2)
fig.set_size_inches((8, 5.5))

for i, fr_idx in enumerate([early_frame_idx, late_frame_idx]):
    img_l, _ = u.read_image_pair(fr_idx)
    axes[i].imshow(img_l, cmap='gray', vmin=0, vmax=255)
    axes[i].axis('off')

    if i == 0:
        inls = [m.back_frame_match.left_keypoint.pt for m in supporters]
        outls = [m.back_frame_match.left_keypoint.pt for m in outliers]
    else:
        inls = [m.front_frame_match.left_keypoint.pt for m in supporters]
        outls = [m.front_frame_match.left_keypoint.pt for m in outliers]
    axes[i].scatter([p[0] for p in inls], [p[1] for p in inls], c="lightblue", s=7.5, marker='o', label='Inlier')
    axes[i].scatter([p[0] for p in outls], [p[1] for p in outls], c="orange", s=7.5, marker='x', label='Outlier')
    axes[i].set_title(f"Frame {fr_idx}")
    axes[0].legend()

plt.show()

##################################
#         Question 5B            #
#    Close Subset of Loops       #
##################################

num_keyframes = len(pg.keyframe_symbols)
max_num_loops = [0, 1, 6, 10, None]

kf_trajectories = np.zeros((len(max_num_loops), num_keyframes, 3))
for i, val in enumerate(max_num_loops):
    print(val)
    pose_graph = PoseGraph(traj_opt.bundles)
    pose_graph.optimize_with_loops(verbose=False)
    for j, (fr_idx, symbol) in enumerate(pose_graph.keyframe_symbols.items()):
        pose = pose_graph._optimized_estimates.atPose3(symbol)
        cam = Camera.from_pose3(fr_idx, pose)
        kf_trajectories[i][j] = cam.calculate_coordinates()


colors = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
plt.clf()
for i, val in enumerate(max_num_loops):
    if val is None:
        l = "All Loops"
    else:
        l = f"First {val} Loops"
    traj = kf_trajectories[i].T  # shape 3xN (N = num keyframes)
    plt.scatter(traj[0], traj[2], marker='o', s=5, c=colors[i], label=l)

plt.scatter(ground_truth_keyframe_trajectory[0], ground_truth_keyframe_trajectory[2],
            marker='o', c='g', s=5, label='GT')
plt.legend()
plt.title("Trajectory while Loop-Closing")
plt.show()


###################################
#         Question 5C             #
#  Location Error & Uncertainty   #
###################################

from logic.trajectory import calculate_trajectory_from_relative_cameras

relative_cameras = pd.read_pickle(os.path.join(c.DATA_WRITE_PATH, 'relative_cameras.pkl'))
full_pnp_trajectory = calculate_trajectory_from_relative_cameras(relative_cameras["Cam_Left"].to_list()).T

pose_graph = PoseGraph(traj_opt.bundles)
pre_closure_marginals = gtsam.Marginals(pose_graph._factor_graph, pose_graph._initial_estimates)
pose_graph.optimize_with_loops(verbose=False)
post_closure_marginals = gtsam.Marginals(pose_graph._factor_graph, pose_graph._optimized_estimates)

first_keyframe_index = min(pose_graph.keyframe_symbols.keys())
first_keyframe_symbol = pose_graph.keyframe_symbols[first_keyframe_index]

num_keyframes = len(pose_graph.keyframe_symbols)
keyframe_pnp_trajectory = np.zeros((num_keyframes, 3))
pre_closure_trajectory = np.zeros((num_keyframes, 3))
pre_closure_uncertainty = np.zeros((num_keyframes,))
post_closure_trajectory = np.zeros((num_keyframes, 3))
post_closure_uncertainty = np.zeros((num_keyframes,))

for i, (fr_idx, symbol) in enumerate(pose_graph.keyframe_symbols.items()):
    keyframe_pnp_trajectory[i] = full_pnp_trajectory[fr_idx]

    # calculate 3D location *before* loop closure:
    pre_pose = pose_graph._initial_estimates.atPose3(symbol)
    pre_cam = Camera.from_pose3(fr_idx, pre_pose)
    pre_closure_trajectory[i] = pre_cam.calculate_coordinates()

    # calculate relative covariance *before* loop closure (relative to first keyframe)
    keys = gtsam.KeyVector()
    keys.append(first_keyframe_symbol)
    keys.append(symbol)

    pre_marginal_cov = pre_closure_marginals.jointMarginalCovariance(keys).fullMatrix()
    pre_info_mat = np.linalg.inv(pre_marginal_cov)
    pre_relative_cov = np.linalg.inv(pre_info_mat[-6:, -6:])
    pre_closure_uncertainty[i] = np.linalg.det(pre_relative_cov)

    # calculate 3D location *after* loop closure:
    post_pose = pose_graph._optimized_estimates.atPose3(symbol)
    post_cam = Camera.from_pose3(fr_idx, post_pose)
    post_closure_trajectory[i] = post_cam.calculate_coordinates()

    # calculate relative covariance *after* loop closure (relative to first keyframe)
    post_marginal_cov = post_closure_marginals.jointMarginalCovariance(keys).fullMatrix()
    post_info_mat = np.linalg.inv(post_marginal_cov)
    post_relative_cov = np.linalg.inv(post_info_mat[-6:, -6:])
    post_closure_uncertainty[i] = np.linalg.det(post_relative_cov)

keyframe_pnp_trajectory = keyframe_pnp_trajectory.T  # shape 3xN
pre_closure_trajectory = pre_closure_trajectory.T    # shape 3xN
post_closure_trajectory = post_closure_trajectory.T   # shape 3xN

plt.clf()
fig = plt.figure(figsize=(12, 8))
gs = plt.GridSpec(nrows=2, ncols=4)

# plot trajectories in main axis:
main_ax = fig.add_subplot(gs[:2, :2])
main_ax.scatter(ground_truth_keyframe_trajectory[0], ground_truth_keyframe_trajectory[2],
                marker='x', c='k', s=5, label='GT')
main_ax.scatter(keyframe_pnp_trajectory[0], keyframe_pnp_trajectory[2],
                marker='o', c="#d73027", s=5, label='PnP')
main_ax.scatter(pre_closure_trajectory[0], pre_closure_trajectory[2],
                marker='o', c="#c51b7d", s=5, label='Bundle Adjustment')
main_ax.scatter(post_closure_trajectory[0], post_closure_trajectory[2],
                marker='o', c="#4575b4", s=5, label='Loop Closure')
main_ax.set_title("Trajectory Estimations")
main_ax.legend()

# plot estimation covariances:
keyframe_indices = [fr_idx for fr_idx in pose_graph.keyframe_symbols.keys()]
lower_right_ax = fig.add_subplot(gs[1, 2:])
lower_right_ax.scatter(keyframe_indices, pre_closure_uncertainty, marker='o', c="#c51b7d", s=2, label='Bundle Adjustment')
lower_right_ax.scatter(keyframe_indices, post_closure_uncertainty, marker='o', c="#4575b4", s=2, label='Loop Closure')
lower_right_ax.set_title("Estimation Uncertainty")

# plot estimation errors:
upper_right_ax = fig.add_subplot(gs[0, 2:], sharex=lower_right_ax)

pnp_euclid_dist = np.linalg.norm(keyframe_pnp_trajectory - ground_truth_keyframe_trajectory, ord=2, axis=0)
pre_closure_euclid_dist = np.linalg.norm(pre_closure_trajectory - ground_truth_keyframe_trajectory, ord=2, axis=0)
post_closure_euclid_dist = np.linalg.norm(post_closure_trajectory - ground_truth_keyframe_trajectory, ord=2, axis=0)

upper_right_ax.scatter(keyframe_indices, pnp_euclid_dist, marker='o', c="#d73027", s=2, label='PnP')
upper_right_ax.scatter(keyframe_indices, pre_closure_euclid_dist, marker='o', c="#c51b7d", s=2, label='Bundle Adjustment')
upper_right_ax.scatter(keyframe_indices, post_closure_euclid_dist, marker='o', c="#4575b4", s=2, label='Loop Closure')
upper_right_ax.set_title("Estimation Errors")

plt.show()

######################################
# plot each figure separately

plt.clf()
plt.scatter(keyframe_indices, pnp_euclid_dist, marker='o', c="#d73027", s=5, label='PnP')
plt.scatter(keyframe_indices, pre_closure_euclid_dist, marker='o', c="#c51b7d", s=5, label='Bundle Adjustment')
plt.scatter(keyframe_indices, post_closure_euclid_dist, marker='o', c="#4575b4", s=5, label='Loop Closure')
plt.title("Estimation Errors")
plt.legend()
plt.show()

plt.clf()
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].scatter(keyframe_indices, pre_closure_uncertainty, marker='o', c="#c51b7d", s=2, label='Bundle Adjustment')
axes[0].set_title("Bundle Adjustment Estimation Uncertainty")
axes[1].scatter(keyframe_indices, post_closure_uncertainty, marker='o', c="#4575b4", s=2, label='Loop Closure')
axes[1].set_title("Loop Closure Estimation Uncertainty")
plt.show()


