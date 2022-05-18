import os
import numpy as np
import pandas as pd
import gtsam

import matplotlib
matplotlib.use("webagg")
from matplotlib import pyplot as plt

import utils as u
from models.camera import Camera
from models.gtsam_frame import GTSAMFrame
from models.database import DataBase
from logic.db_adapter import DBAdapter


###############################
#        PREPROCESSING        #
#     read data from EX04     #
###############################

# need to read DFs from WSL's filesystem because the default directories are Windows fs
wsl_write_path = "/mnt/c/Users/nirjo/Documents/University/Masters/Computer Vision Aided Navigation/VAN_ex/docs/db"
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
# TODO: should this be it's own class?

# long_track_idx, _ = dba.sample_track_idx_with_length(min_len=10, max_len=15)  # for simplicity, don't allow too-long tracks

# TODO: DELETE THIS -- example long_track_idx: 196061
long_track_data = dba.tracks_db.xs(196061, level=DataBase.TRACKIDX)

# triangulate the landmark to a 3D point using the last Frame's pixels (and gtsam):
# long_track_data = dba.tracks_db.xs(long_track_idx, level=DataBase.TRACKIDX)
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

    # store initial_estimates:
    errs_dict[frame_idx] = (left_err, right_err, factor_err)

# plot errors:
errs_df = pd.DataFrame.from_dict(errs_dict, orient='index', columns=["left_error", "right_error", "factor_error"])
plt.clf()
fig, axes = plt.subplots(1, 2)
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







