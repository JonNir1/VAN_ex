import gtsam
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from models.camera import Camera
from models.gtsam_frame import GTSAMFrame
from models.database import DataBase
from logic.db_adapter import DBAdapter


class Bundle:
    __min_size = 5
    __max_size = 20

    NoiseModel = gtsam.noiseModel.Isotropic.Sigma(3, 1)

    def __init__(self, tracks: pd.DataFrame, cameras: pd.DataFrame):
        # TODO: assert correct size for Bundle
        self.initial_estimates = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        self._build_bundle(tracks, cameras)

    def adjust(self):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates)
        return optimizer.optimize()

    def _build_bundle(self, tracks: pd.DataFrame, cameras: pd.DataFrame):
        # iterates over tracks and cameras and builds the Bundles members ($initial_estimate and $graph)
        gtsam_frames = self._preprocess_frames(cameras)
        self._preprocess_tracks(tracks, gtsam_frames)

    def _preprocess_frames(self, cameras_df: pd.DataFrame) -> Dict[int, GTSAMFrame]:
        frames_dict = {}
        for frame_idx in cameras_df.index:
            left_camera, _ = cameras_df.xs(frame_idx)
            gtsam_frame = GTSAMFrame.from_camera(left_camera)
            frames_dict[frame_idx] = gtsam_frame
            # add initial est. of the camera position:
            self.initial_estimates.insert(gtsam_frame.symbol, gtsam_frame.pose)

            # for the first Frame in the Bundle, add a PriorFactor to the graph
            # TODO: use previous Bundle's output to add prior to current Bundle
            if frame_idx == cameras_df.index.min():
                pose_factor = gtsam.PriorFactorPose3(gtsam_frame.symbol, gtsam_frame.pose, self.NoiseModel)
                self.graph.add(pose_factor)
        return frames_dict

    def _preprocess_tracks(self, tracks: pd.DataFrame, gtsam_frames: Dict[int, GTSAMFrame]):
        for tr_idx in tracks.index.unique(level=DataBase.TRACKIDX):
            # add landmark estimation for the Track's 3D landmark
            single_track_data = tracks.xs(tr_idx, level=DataBase.TRACKIDX)
            landmark_3D = self.__calculate_landmark(single_track_data, gtsam_frames)
            landmark_symbol = gtsam.symbol('l', tr_idx)
            self.initial_estimates.insert(landmark_symbol, landmark_3D)  # add initial est. of the landmark position

            # add projection factor for each Frame participating in this Track
            for fr_idx in single_track_data.index:
                x_l, x_r, y = single_track_data.xs(fr_idx)
                stereo_point2D = gtsam.StereoPoint2(x_l, x_r, y)
                camera_symbol, _, stereo_params = gtsam_frames[fr_idx]
                factor = gtsam.GenericStereoFactor3D(stereo_point2D, self.NoiseModel, camera_symbol, landmark_symbol, stereo_params)
                self.graph.add(factor)

    @staticmethod
    def __calculate_landmark(track_data: pd.DataFrame, gtsam_frames: Dict[int, GTSAMFrame]) -> gtsam.Point3:
        last_frame_idx = track_data.index.max()
        _, pose, stereo_params = gtsam_frames[last_frame_idx]
        stereo_cameras = gtsam.StereoCamera(pose, stereo_params)
        x_l, x_r, y = track_data.xs(last_frame_idx)
        landmark = stereo_cameras.backproject(gtsam.StereoPoint2(x_l, x_r, y))
        return landmark

