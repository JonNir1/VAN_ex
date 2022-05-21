import os
import numpy as np
import pandas as pd
import gtsam

import matplotlib
from matplotlib import pyplot as plt
from gtsam.utils.plot import plot_trajectory, plot_3d_points

import utils as u
from models.camera import Camera
from models.gtsam_frame import GTSAMFrame
from models.database import DataBase
from logic.db_adapter import DBAdapter
from logic.gtsam_bundle import Bundle
from service.trajectory_optimizer import TrajectoryOptimizer


###############################
#        PREPROCESSING        #
#     read data from EX04     #
###############################

# change matplotlib's backend
matplotlib.use("webagg")

# need to read DFs from WSL's filesystem because the default directories are Windows fs
wsl_base_path = "/mnt/c/Users/nirjo/Documents/University/Masters/Computer Vision Aided Navigation/VAN_ex"
wsl_read_path = wsl_base_path + "/dataset"
wsl_write_path = wsl_base_path + "/docs/db"
dba = DBAdapter(data=[])
dba.tracks_db = pd.read_pickle(os.path.join(wsl_write_path, 'tracksdb_02052022_2012.pkl'))
dba.cameras_db = pd.read_pickle(os.path.join(wsl_write_path, 'camerasdb_02052022_2012.pkl'))

# need to read camera matrices manually, which is very ugly but necessary because of WSL :(
first_cam_path = "/mnt/c/Users/nirjo/Documents/University/Masters/Computer Vision Aided Navigation/VAN_ex/dataset/sequences/00/calib.txt"
K, _, Mright = u.read_first_camera_matrices(first_cam_path)
Camera._K = K
Camera._RightRotation = Mright[:, :3]
Camera._RightTranslation = Mright[:, 3:]


########################################
#             Question 5.1             #
#          reprojection error          #
########################################

# example long_track_idx: 196061
long_track_idx, _ = dba.sample_track_idx_with_length(min_len=10, max_len=15)  # for simplicity, don't allow too-long tracks

# triangulate the landmark to a 3D point using the last Frame's pixels (and gtsam):
long_track_data = dba.tracks_db.xs(long_track_idx, level=DataBase.TRACKIDX)
long_track_frame_idxs = long_track_data.index.to_list()
left_last_cam, _ = dba.cameras_db.xs(max(long_track_frame_idxs))
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
for frame_idx in reversed(long_track_frame_idxs):
    # project the 3D point on the camera:
    left_camera, _ = dba.cameras_db.xs(frame_idx)
    gtsam_camera = GTSAMFrame.from_camera(left_camera)
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

# plot errors:
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
first_bundle_cameras = dba.cameras_db[dba.cameras_db.index.get_level_values(level=DataBase.FRAMEIDX).isin(np.arange(15))]
first_bundle_gtsam_frames = first_bundle_cameras["Cam_Left"].apply(lambda cam: GTSAMFrame.from_camera(cam))
first_bundle_landmark_symbols = pd.Series(first_bundle_tracks.index.unique(level=DataBase.TRACKIDX)).apply(
    lambda tr_idx: gtsam.symbol('l', tr_idx))

# create the Bundle and optimize
first_bundle = Bundle(first_bundle_gtsam_frames, first_bundle_tracks, first_bundle_landmark_symbols)
print(f"Error before optimization: {first_bundle.error:.10f}")
result = first_bundle.adjust()
print(f"Error after optimization: {first_bundle.error:.10f}")

# plot 3D and 2D scenes:
plt.clf()
plot_trajectory(fignum=0, values=result, title="First Bundle Trajectory")
plot_3d_points(fignum=1, values=result, title="First Bundle Landmarks")
plt.show()


#########################################
#            Question 5.3               #
#         full bundle adjusted          #
#########################################

long_tracks_db = dba.prune_short_tracks(4)  # only use tracks of length 4 or longer
trajectory_optimizer = TrajectoryOptimizer(cameras=dba.cameras_db, tracks=long_tracks_db)
bundle_results = trajectory_optimizer.optimize(verbose=True)

left_cameras = trajectory_optimizer.extract_cameras(bundle_results)
landmarks_3d = trajectory_optimizer.extract_landmarks(bundle_results)

# filter landmarks with irrelevant X,Z coordinates
minX, maxX = -200, 400
minZ, maxZ = -100, 500
x_in_range = np.bitwise_and(minX <= landmarks_3d[0], landmarks_3d[0] <= maxX)
z_in_range = np.bitwise_and(minZ <= landmarks_3d[2], landmarks_3d[2] <= maxZ)
filtered_landmarks_3d = landmarks_3d[:, np.bitwise_and(x_in_range, z_in_range)]

# need to read ground truth with filepath matching WSL's file system:
poses_path = wsl_read_path + '/poses/00.txt'
poses_file = open(poses_path, 'r')
real_trajectory = np.zeros((3450, 3))
for i, line in enumerate(poses_file.readlines()):
    mat = np.array(line.split(), dtype=float).reshape((3, 4))
    R = mat[:, :3]
    t = mat[:, 3:]
    real_trajectory[i] -= (R.T @ t).reshape((3,))
poses_file.close()

# plot all (left) cameras to see the complete trajectory (NOT part of H.W. requests):
plt.clf()
estimated_trajectory = np.zeros((3450, 3))
optimized_trajectory = np.zeros((3450, 3))
for i in range(3450):
    estimated_cam = dba.cameras_db["Cam_Left"].xs(i)
    estimated_trajectory[i] = estimated_cam.calculate_coordinates()
    optimized_cam = left_cameras[i]
    optimized_trajectory[i] = optimized_cam.calculate_coordinates()

plt.scatter(estimated_trajectory.T[0], estimated_trajectory.T[2], marker="o", c='b', s=1, label="estimated (PnP)")
plt.scatter(optimized_trajectory.T[0], optimized_trajectory.T[2], marker="x", s=1, c='r', label="optimized (GTSAM)")
plt.scatter(real_trajectory.T[0], real_trajectory.T[2], marker="*", s=1, c='k', label="ground truth")
plt.legend()
plt.show()

# present view from above of keyframes & landmarks
# as well as Euclidean distance between optimized keyframe location and ground-truth locations
optimized_keyframe_positions = optimized_trajectory.take(indices=trajectory_optimizer.keyframe_indices, axis=0)
real_keyframe_positions = real_trajectory.take(indices=trajectory_optimizer.keyframe_indices, axis=0)
euclidean_distances = np.linalg.norm(optimized_keyframe_positions - real_keyframe_positions, ord=2, axis=1)

plt.clf()
fig, axes = plt.subplots(1, 2)
axes[0].scatter(filtered_landmarks_3d[0], filtered_landmarks_3d[2], marker='*', c='orange', s=0.5, label='landmark')
axes[0].scatter(optimized_keyframe_positions.T[0], optimized_keyframe_positions.T[2],
                marker='o', c='r', s=5, label='estimated (GTSAM)')
axes[0].scatter(real_keyframe_positions.T[0], real_keyframe_positions.T[2],
                marker='o', c='b', s=5, label='ground truth')
axes[0].set_title("KITTI Trajectories")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Z")
axes[0].legend(loc='best')

axes[1].scatter(trajectory_optimizer.keyframe_indices, euclidean_distances, c='k', marker='o', s=2)
axes[1].set_title("Euclidean Distances")
axes[1].set_xlabel("Frame ID")
fig.set_figwidth(15)
plt.show()
