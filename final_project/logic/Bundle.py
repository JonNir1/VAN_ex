import time
import gtsam
import numpy as np
import pandas as pd
from typing import List

import final_project.config as c
import final_project.camera_utils as cu
from final_project.models.Camera import Camera
from final_project.models.FactorGraph import FactorGraph


class Bundle:
    _PixelCovariance = 1
    _LocationCovariance = 0.01
    _AngleCovariance = (0.1 * np.pi / 180) ** 2
    _Max_Landmark_Distance = 400
    PointNoiseModel = gtsam.noiseModel.Isotropic.Sigma(3, _PixelCovariance)
    PoseNoiseModel = gtsam.noiseModel.Diagonal.Sigmas(np.array([_AngleCovariance, _AngleCovariance, _AngleCovariance,
                                                                _LocationCovariance, _LocationCovariance,
                                                                _LocationCovariance]))

    def __init__(self, tracks: pd.DataFrame, cameras: pd.DataFrame, verbose=False):
        self._factor_graph = FactorGraph()
        self._cameras = self.__process_cameras(cameras, verbose)  # cols: CamL, Symbol, AbsPose, InitPose, OptPose
        # self._tracks = self.__process_tracks(tracks, verbose)  # cols: Xl, Xr, Y, Symbol

    @property
    def is_optimized(self) -> bool:
        return self._factor_graph.is_optimized

    def adjust(self):
        self._factor_graph.optimize()
        # update poses in the bundle
        opt_poses = []
        for i, row in self._cameras.iterrows():
            symbol = row[c.Symbol]
            pose = self._factor_graph.get_optimized_pose(symbol)
            opt_poses.append(pose)
        self._cameras.loc[:, c.OptPose] = opt_poses

    def __process_cameras(self, cameras: pd. DataFrame, verbose):
        start, min_count = time.time(), 0
        first_abs_pose = cameras.iloc[0][c.AbsolutePose]
        relative_poses = []
        for i, row in cameras.iterrows():
            symbol = row[c.Symbol]
            abs_pose = row[c.AbsolutePose]
            rel_pose = first_abs_pose.between(abs_pose)
            relative_poses.append(rel_pose)
            self._factor_graph.add_initial_estimate(symbol, rel_pose)
            if i == 0:
                # add Prior for the first frame in the Bundle
                prior_factor = gtsam.PriorFactorPose3(symbol, rel_pose, self.PoseNoiseModel)
                self._factor_graph.add_factor(prior_factor)

            curr_minute = int((time.time() - start) / 60)
            if verbose and curr_minute > min_count:
                print(f"\tfinished {i} Cameras in {curr_minute} minutes")
        init_pose, opt_pose = c.InitialPose, c.OptPose
        cameras = cameras.assign(init_pose=relative_poses, opt_pose=None)

        if verbose:
            elapsed = time.time() - start
            print(f"\tfinished {i} Cameras in {elapsed:.2f} seconds")
        return cameras

    def __process_tracks(self, tracks: pd.DataFrame, verbose):
        start, min_count = time.time(), 0
        stereo_params = cu.calculate_gtsam_stereo_params()
        counter = 0
        for track_idx in tracks.index.unique(level=c.TrackIdx):
            single_track_data = tracks.xs(track_idx, level=c.TrackIdx)

            # add initial est. of the landmark position
            last_frame_idx = single_track_data.index.max()
            x_l, x_r, y, _ = single_track_data.xs(last_frame_idx)
            landmark_symbol = single_track_data.loc[last_frame_idx, c.Symbol]  # need to extract symbol separately otherwise it is considered a float
            pose = self._cameras.loc[last_frame_idx, c.InitialPose]
            stereo_cameras = gtsam.StereoCamera(pose, stereo_params)
            landmark_3D = stereo_cameras.backproject(gtsam.StereoPoint2(x_l, x_r, y))
            if landmark_3D[2] <= 0 or landmark_3D[2] >= self._Max_Landmark_Distance:
                # the Z coordinate of the landmark is behind the camera or too distant
                # do not include this landmark in the bundle
                continue
            self._factor_graph.add_initial_estimate(landmark_symbol, landmark_3D)

            # add projection factor for each Frame participating in this Track
            for fr_idx in single_track_data.index:
                x_l, x_r, y, _ = single_track_data.xs(fr_idx)
                stereo_point2D = gtsam.StereoPoint2(x_l, x_r, y)
                camera_symbol = self._cameras.loc[fr_idx, c.Symbol]
                factor = gtsam.GenericStereoFactor3D(stereo_point2D, self.PointNoiseModel, camera_symbol,
                                                     landmark_symbol, stereo_params)
                self._factor_graph.add_factor(factor)

            counter += 1
            curr_minute = int((time.time() - start) / 60)
            if verbose and curr_minute > min_count:
                print(f"\tfinished {counter} Tracks in {curr_minute} minutes")

        if verbose:
            elapsed = time.time() - start
            print(f"\tfinished {counter} Tracks in {elapsed:.2f} seconds")
        return tracks





