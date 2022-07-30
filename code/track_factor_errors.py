import os
import time
import gtsam
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

import config as c
import utils as u
from models.directions import Side
from models.camera import Camera
from service.frame_processor import FrameProcessor
from logic.db_adapter import DBAdapter
from models.database import DataBase
from models.gtsam_frame import GTSAMFrame
from service.TrajectoryOptimizer2 import TrajectoryOptimizer2
from logic.PoseGraph import PoseGraph

# change matplotlib's backend
matplotlib.use("webagg")

#####################################

# need to read camera matrices manually, which is very ugly but necessary because of WSL :(
first_cam_path = "/mnt/c/Users/nirjo/Documents/University/Masters/Computer Vision Aided Navigation/VAN_ex/dataset/sequences/00/calib.txt"
K, _, Mright = u.read_first_camera_matrices(first_cam_path)
Camera._K = K
Camera._RightRotation = Mright[:, :3]
Camera._RightTranslation = Mright[:, 3:]

dba = DBAdapter(data=[])
dba.tracks_db = pd.read_pickle(os.path.join(c.DATA_WRITE_PATH, 'tracksdb_28072022_1049.pkl'))
dba.cameras_db = pd.read_pickle(os.path.join(c.DATA_WRITE_PATH, 'camerasdb_28072022_1049.pkl'))
traj_opt = TrajectoryOptimizer2(tracks=dba.tracks_db, relative_cams=dba.cameras_db[DataBase.CAM_LEFT])
traj_opt.optimize(verbose=True)

#####################################

# init stereo params for gtsam:
cam_0_l = dba.cameras_db.loc[0, DataBase.CAM_LEFT]
gtsam_frame0 = GTSAMFrame.from_camera(cam_0_l)

relevant_tracks = traj_opt.tracks  # maybe take a subset of tracks

pre_optimization_errors = {k: [] for k in range(65)}
post_optimization_errors = {k: [] for k in range(65)}
for i, track_id in enumerate(relevant_tracks.index.unique(level=DataBase.TRACKIDX)):
    if i % 10000 == 0:
        print(f"finished {i} tracks")
    is_track_in_bundle = [track_id in b.landmark_symbols.keys() for b in traj_opt.bundles]
    if not any(is_track_in_bundle):
        # skip tracks that were filtered out of all bundles
        continue
    last_bundle_id = np.argmax(is_track_in_bundle)
    track_data = relevant_tracks[relevant_tracks.index.get_level_values(level=DataBase.TRACKIDX) == track_id]
    track_frame_ids = track_data.index.unique(level=DataBase.FRAMEIDX)
    last_frame_id = track_frame_ids.max()
    for b in traj_opt.bundles:
        if track_id not in b.landmark_symbols.keys():
            continue
        landmark_symbol = b.landmark_symbols[track_id]
        for fr_idx in track_frame_ids:
            if fr_idx not in b.frame_symbols.keys():
                continue
            frame_distance = last_frame_id - fr_idx
            fr_symbol = b.frame_symbols[fr_idx]
            x_l, x_r, y = track_data.xs((fr_idx, track_id))
            stereo_point2D = gtsam.StereoPoint2(x_l, x_r, y)
            factor = gtsam.GenericStereoFactor3D(stereo_point2D, b.PointNoiseModel, fr_symbol,
                                                 landmark_symbol, gtsam_frame0.stereo_params)
            pre_err = factor.error(b.initial_estimates)
            pre_optimization_errors[frame_distance].append(pre_err)
            post_err = factor.error(b.optimized_estimates)
            post_optimization_errors[frame_distance].append(post_err)

median_pre_errors = {k: pd.Series(v).median() for k, v in pre_optimization_errors.items()}
median_post_errors = {k: pd.Series(v).median() for k, v in post_optimization_errors.items()}
median_errors = pd.DataFrame([median_pre_errors, median_post_errors]).T
median_errors.columns = ["Pre", "Post"]
median_errors.to_pickle(os.path.join(c.DATA_WRITE_PATH, "track_factor_errors.pkl"))


########  same measure, only WITHIN each bundle  ########

pre_optimization_errors2 = {k: [] for k in range(15)}
post_optimization_errors2 = {k: [] for k in range(15)}
for i, b in enumerate(traj_opt.bundles):
    if i%10 == 0:
        print(f"Starting bundle #{i}\n")
    bundle_frame_idxs = list(b.frame_symbols.keys())
    bundle_track_idxs = list(b.landmark_symbols.keys())
    bundle_tracks = traj_opt.tracks[traj_opt.tracks.index.get_level_values(level=DataBase.FRAMEIDX).isin(bundle_frame_idxs)]
    for track_idx in bundle_track_idxs:
        single_track_data = bundle_tracks.xs(track_idx, level=DataBase.TRACKIDX)
        landmark_symbol = b.landmark_symbols[track_idx]

        last_frame_idx = single_track_data.index.max()
        for fr_idx in single_track_data.index:
            frame_distance = last_frame_idx - fr_idx
            fr_symbol = b.frame_symbols[fr_idx]
            x_l, x_r, y = single_track_data.xs(fr_idx)
            stereo_point2D = gtsam.StereoPoint2(x_l, x_r, y)
            factor = gtsam.GenericStereoFactor3D(stereo_point2D, b.PointNoiseModel, fr_symbol,
                                                 landmark_symbol, gtsam_frame0.stereo_params)
            pre_err = factor.error(b.initial_estimates)
            pre_optimization_errors2[frame_distance].append(pre_err)
            post_err = factor.error(b.optimized_estimates)
            post_optimization_errors2[frame_distance].append(post_err)


median_pre_errors2 = {k: pd.Series(v).median() for k, v in pre_optimization_errors2.items()}
median_post_errors2 = {k: pd.Series(v).median() for k, v in post_optimization_errors2.items()}
median_errors2 = pd.DataFrame([median_pre_errors2, median_post_errors2]).T
median_errors2.columns = ["Pre", "Post"]

