import gtsam
import numpy as np
import pandas as pd

from models.database import DataBase


class Bundle:

    PointNoiseModel = gtsam.noiseModel.Isotropic.Sigma(3, 1)
    PoseNoiseModel = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.pi / 180, np.pi / 180, np.pi / 180, 1, 1, 1]))

    def __init__(self, gtsam_frames: pd.Series, tracks_data: pd.DataFrame, landmark_symbols: pd.Series):
        # TODO: assert correct size of Bundle (5-20 frames)
        self.estimates = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        self._process_frames(gtsam_frames)
        self._process_tracks(gtsam_frames, tracks_data, landmark_symbols)

    @property
    def error(self):
        return self.graph.error(self.estimates)

    def adjust(self):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.estimates)
        self.estimates = optimizer.optimize()
        return self.estimates

    def _process_frames(self, gtsam_frames: pd.Series):
        # add camera poses to initial estimates:
        gtsam_frames.apply(lambda gt_fr: self.estimates.insert(gt_fr.symbol, gt_fr.pose))
        # add a PriorFactor for the first Frame in the Bundle:
        first_frame_in_bundle = gtsam_frames[gtsam_frames.index.min()]
        pose_factor = gtsam.PriorFactorPose3(first_frame_in_bundle.symbol, first_frame_in_bundle.pose, self.PoseNoiseModel)
        self.graph.add(pose_factor)

    def _process_tracks(self, gtsam_frames: pd.Series, tracks_data: pd.DataFrame, landmark_symbols: pd.Series):
        for track_idx in tracks_data.index.unique(level=DataBase.TRACKIDX):
            # add landmark estimation for the Track's 3D landmark
            single_track_data = tracks_data.xs(track_idx, level=DataBase.TRACKIDX)
            last_frame_idx = single_track_data.index.max()
            _, pose, stereo_params = gtsam_frames[last_frame_idx]
            stereo_cameras = gtsam.StereoCamera(pose, stereo_params)
            x_l, x_r, y = single_track_data.xs(last_frame_idx)
            landmark_3D = stereo_cameras.backproject(gtsam.StereoPoint2(x_l, x_r, y))
            landmark_symbol = landmark_symbols.xs(track_idx)
            self.estimates.insert(landmark_symbol, landmark_3D)  # add initial est. of the landmark position

            # add projection factor for each Frame participating in this Track
            for fr_idx in single_track_data.index:
                x_l, x_r, y = single_track_data.xs(fr_idx)
                stereo_point2D = gtsam.StereoPoint2(x_l, x_r, y)
                camera_symbol, _, stereo_params = gtsam_frames[fr_idx]
                factor = gtsam.GenericStereoFactor3D(stereo_point2D, self.PointNoiseModel, camera_symbol,
                                                     landmark_symbol, stereo_params)
                self.graph.add(factor)
        return

