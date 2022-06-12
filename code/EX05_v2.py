import os
import numpy as np
import pandas as pd
import gtsam

import matplotlib
from matplotlib import pyplot as plt
from gtsam.utils.plot import plot_trajectory, plot_3d_points

import config as c
import utils as u
from models.directions import Side
from models.camera import Camera
from models.gtsam_frame import GTSAMFrame
from models.database import DataBase
from logic.db_adapter import DBAdapter
from logic.Bundle2 import Bundle2
from logic.trajectory import calculate_trajectory_from_relative_cameras, read_ground_truth_trajectory
from service.TrajectoryOptimizer2 import TrajectoryOptimizer2

# change matplotlib's backend
matplotlib.use("webagg")

##################################
#          Load Data             #
##################################

dba = DBAdapter(data=[])
dba.tracks_db = pd.read_pickle(os.path.join(c.DATA_WRITE_PATH, 'tracksdb_02052022_2012.pkl'))
dba.cameras_db = pd.read_pickle(os.path.join(c.DATA_WRITE_PATH, 'camerasdb_02052022_2012.pkl'))

# need to read camera matrices manually, which is very ugly but necessary because of WSL :(
first_cam_path = "/mnt/c/Users/nirjo/Documents/University/Masters/Computer Vision Aided Navigation/VAN_ex/dataset/sequences/00/calib.txt"
K, _, Mright = u.read_first_camera_matrices(first_cam_path)
Camera._K = K
Camera._RightRotation = Mright[:, :3]
Camera._RightTranslation = Mright[:, 3:]

# read ground truth with filepath matching WSL's file system:
wsl_base_path = "/mnt/c/Users/nirjo/Documents/University/Masters/Computer Vision Aided Navigation/VAN_ex"
wsl_read_path = wsl_base_path + "/dataset"
poses_path = wsl_read_path + '/poses/00.txt'
poses_file = open(poses_path, 'r')
real_trajectory = np.zeros((3450, 3))
for i, line in enumerate(poses_file.readlines()):
    mat = np.array(line.split(), dtype=float).reshape((3, 4))
    R = mat[:, :3]
    t = mat[:, 3:]
    real_trajectory[i] -= (R.T @ t).reshape((3,))
poses_file.close()
real_trajectory = real_trajectory.T

####################################
#         Pre-Processing           #
#   1. Calculate Relative Cameras  #
#   2. Drop Short Tracks           #
####################################

relative_cams = []
for i, cam in enumerate(dba.cameras_db["Cam_Left"]):
    if i == 0:
        relative_cams.append(cam)
        continue
    back_cam = dba.cameras_db["Cam_Left"][i-1]
    back_R, back_t = back_cam.get_rotation_matrix(), back_cam.get_translation_vector()
    front_R, front_t = cam.get_rotation_matrix(), cam.get_translation_vector()
    R = front_R @ back_R.T
    t = front_t - R @ back_t
    ext = np.hstack([R, t])
    rel_cam = Camera(i, Side.LEFT, ext)
    relative_cams.append(rel_cam)

left_cams = pd.Series(relative_cams, name="Cam_Left")
right_cams = left_cams.apply(lambda cam: cam.calculate_right_camera())
right_cams.name = "Cam_Right"
relative_cameras_db = pd.DataFrame([left_cams, right_cams]).T
relative_cameras_db.index.name = DataBase.FRAMEIDX

long_tracks_db = dba.prune_short_tracks(4)  # only use tracks of length 4 or longer


dba.tracks_db = long_tracks_db
dba.cameras_db = relative_cameras_db


########################################
#             Question 5.1             #
#          reprojection error          #
########################################

long_track_idx, _ = dba.sample_track_idx_with_length(min_len=10, max_len=15)  # for simplicity, don't allow too-long tracks
long_track_idx: 196061  # example

# triangulate the landmark to a 3D point using the last Frame's pixels (and gtsam):
long_track_data = dba.tracks_db.xs(long_track_idx, level=DataBase.TRACKIDX)
long_track_frame_idxs = long_track_data.index.to_list()

# create cameras relative to track's first camera
track_cameras = []
for i, fr_idx in enumerate(long_track_frame_idxs):
    if i == 0:
        R, t = np.eye(3), np.zeros((3, 1))
        track_cameras.append(Camera(i, Side.LEFT, np.hstack([R, t])))
    else:
        prev_cam = track_cameras[-1]
        prev_R, prev_t = prev_cam.get_rotation_matrix(), prev_cam.get_translation_vector()
        curr_cam_rel = dba.cameras_db.loc[fr_idx, "Cam_Left"]
        curr_R_rel, curr_t_rel = curr_cam_rel.get_rotation_matrix(), curr_cam_rel.get_translation_vector()
        curr_R = curr_R_rel @ prev_R
        curr_t = curr_t_rel + curr_R_rel @ prev_t
        track_cameras.append(Camera(i, Side.LEFT, np.hstack([curr_R, curr_t])))

left_last_cam = track_cameras[-1]
gtsam_last_camera = GTSAMFrame.from_camera(left_last_cam)
last_stereo_cams = gtsam.StereoCamera(gtsam_last_camera.pose, gtsam_last_camera.stereo_params)
x_l, x_r, y = long_track_data.xs(max(long_track_frame_idxs))
landmark_3D = last_stereo_cams.backproject(gtsam.StereoPoint2(x_l, x_r, y))

# init gtsam symbols:
noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1)
l1 = gtsam.symbol('l', 1)
frame_symbols = {fr_idx : gtsam.symbol('c', fr_idx) for fr_idx in long_track_frame_idxs}

# calculate Euclidean distances (L2 norm) and factor error of the re-projected landmark:
errs_dict = dict()
for i in reversed(range(len(long_track_frame_idxs))):
    frame_idx = long_track_frame_idxs[i]

    cam = track_cameras[i]
    gtsam_camera = GTSAMFrame.from_camera(cam)
    gtsam_stereo_cameras = gtsam.StereoCamera(gtsam_camera.pose, gtsam_camera.stereo_params)
    projected_landmark = gtsam_stereo_cameras.project(landmark_3D)
    proj_x_l, proj_x_r, proj_y = projected_landmark.vector()

    # calculate Euclidean distance:
    x_l, x_r, y = long_track_data.xs(frame_idx)
    left_err = np.linalg.norm(np.array([x_l, y]) - np.array([proj_x_l, proj_y]), ord=2)
    right_err = np.linalg.norm(np.array([x_r, y]) - np.array([proj_x_r, proj_y]), ord=2)

    # calculate factor error:
    gtsam_stereo_point = gtsam.StereoPoint2(x_l, x_r, y)
    camera_symbol = frame_symbols[frame_idx]
    factor = gtsam.GenericStereoFactor3D(gtsam_stereo_point, noise_model, camera_symbol, l1, gtsam_camera.stereo_params)
    initial_estimate = gtsam.Values()
    initial_estimate.insert(l1, gtsam.Point3(landmark_3D))
    initial_estimate.insert(camera_symbol, gtsam_camera.pose)
    factor_err = factor.error(initial_estimate)

    # store values:
    errs_dict[frame_idx] = (left_err, right_err, factor_err)

# plot the errors:
errs_df = pd.DataFrame.from_dict(errs_dict, orient='index', columns=["left_error", "right_error", "factor_error"])
plt.clf()
_, axes = plt.subplots(1, 2)
axes[0].scatter(errs_df.index, errs_df["left_error"], label="Euclidean Distance", c='b')
axes[0].scatter(errs_df.index, errs_df["factor_error"], label="Factor Error", c='r')
axes[0].set_title("Re-Projection Errors")
axes[0].set_xlabel("FrameIdx")
axes[0].legend()

axes[1].scatter(errs_df["left_error"], errs_df["factor_error"], c='r')
axes[1].set_title("Factor Errors for Euclidean Distance")
axes[1].set_xlabel("Euclidean Distance")
axes[1].set_ylabel("Factor Error")
plt.show()

##########################################
#             Question 5.2               #
#         first bundle adjusted          #
##########################################

# collect all tracks & cameras related to first 15 Frames:
first_bundle_tracks = dba.tracks_db[dba.tracks_db.index.get_level_values(level=DataBase.FRAMEIDX).isin(np.arange(15))]
first_bundle_camera_db = dba.cameras_db[dba.cameras_db.index.get_level_values(level=DataBase.FRAMEIDX).isin(np.arange(15))]
first_bundle_cameras = first_bundle_camera_db[DataBase.CAM_LEFT].to_list()

# create Bundle & optimize:
first_bundle = Bundle2(first_bundle_cameras, first_bundle_tracks)
print(f"Error before optimization: {first_bundle.calculate_error(first_bundle.initial_estimates):.5f}")
first_bundle.adjust()
print(f"Error after optimization: {first_bundle.calculate_error(first_bundle.optimized_estimates):.5f}")

# plot 3D and 2D scenes:
plt.clf()
plot_trajectory(fignum=0, values=first_bundle.optimized_estimates, title="First Bundle Trajectory")
plot_3d_points(fignum=1, values=first_bundle.optimized_estimates, title="First Bundle Landmarks")
plt.show()

bundle_track = calculate_trajectory_from_relative_cameras(first_bundle.extract_relative_cameras())
bundle_gt = real_trajectory[:, :15]
euclidean_distances = np.linalg.norm(bundle_track - bundle_gt, ord=2, axis=0)

plt.clf()
plt.scatter(bundle_track[0], bundle_track[2], marker="x", s=1, c='g')
plt.scatter(bundle_gt[0], bundle_gt[2], marker="*", s=1, c='k', label="ground truth")
plt.show()

#########################################
#            Question 5.3               #
#         full bundle adjusted          #
#########################################

traj_opt = TrajectoryOptimizer2(tracks=dba.tracks_db, relative_cams=dba.cameras_db[DataBase.CAM_LEFT])
traj_opt.optimize(verbose=True)
left_cams = traj_opt.extract_all_relative_cameras()

optimized_trajectory = calculate_trajectory_from_relative_cameras(left_cams)

# plot G.T. and optimized trajectories
plt.clf()
plt.scatter(optimized_trajectory[0], optimized_trajectory[2], marker="x", s=1, c='g', label="optimized (GTSAM)")
plt.scatter(real_trajectory[0], real_trajectory[2], marker="*", s=1, c='k', label="ground truth")
plt.legend()
plt.show()

# present view from above of keyframes & landmarks
# as well as Euclidean distance between optimized keyframe location and ground-truth locations
optimized_keyframe_positions = optimized_trajectory.take(indices=traj_opt.keyframe_idxs, axis=1)
real_keyframe_positions = real_trajectory.take(indices=traj_opt.keyframe_idxs, axis=1)
euclidean_distances = np.linalg.norm(optimized_keyframe_positions - real_keyframe_positions, ord=2, axis=0)

plt.clf()
fig, axes = plt.subplots(1, 2)
# axes[0].scatter(filtered_landmarks_3d[0], filtered_landmarks_3d[2], marker='*', c='orange', s=0.5, label='landmark')
axes[0].scatter(optimized_keyframe_positions[0], optimized_keyframe_positions[2],
                marker='o', c='r', s=5, label='estimated (GTSAM)')
axes[0].scatter(real_keyframe_positions[0], real_keyframe_positions[2],
                marker='o', c='b', s=5, label='ground truth')
axes[0].set_title("KITTI Trajectories")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Z")
axes[0].legend(loc='best')

axes[1].scatter(traj_opt.keyframe_idxs, euclidean_distances, c='k', marker='o', s=2)
axes[1].set_title("Euclidean Distances")
axes[1].set_xlabel("Frame ID")
fig.set_figwidth(15)
plt.show()

