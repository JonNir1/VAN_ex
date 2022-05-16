import gtsam
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
        self.values = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()


        self.cameras = self._preprocess_frames(cameras)
        self.tracks = tracks




    def adjust(self):
        # TODO
        return




    def _preprocess(self, tracks: pd.DataFrame, cameras: pd.DataFrame):
        gtsam_frames = self._preprocess_frames(cameras)
        return

    def _preprocess_frames(self, cameras_df: pd.DataFrame) -> Dict[int, GTSAMFrame]:
        frames_dict = {}
        for frame_idx in cameras_df.index:
            left_camera, _ = cameras_df.xs(frame_idx)
            gtsam_frame = GTSAMFrame.from_camera(left_camera)
            frames_dict[frame_idx] = gtsam_frame
            self.values.insert(gtsam_frame.symbol, gtsam_frame.pose)
        return frames_dict

    def _preprocess_tracks(self, tracks: pd.DataFrame, cameras_data: Dict[int, GTSAMFrame]):
        for tr_idx in tracks.index.unique(level=DataBase.TRACKIDX):
            landmark_symbol = gtsam.symbol('l', tr_idx)

        return

    def __preprocess_single_track(self, track_idx: int):
        landmark_symbol = gtsam.symbol('l', track_idx)
        landmark_point3D = self.__calculate_landmark(track_idx)
        self.values.insert(landmark_symbol, landmark_point3D)

        track_data = self.tracks.xs(track_idx, level=DataBase.TRACKIDX)
        for fr_idx in track_data.index:
            x_l, x_r, y = track_data.xs(fr_idx)
            stereo_point2D = gtsam.StereoPoint2(x_l, x_r, y)
            camera_symbol, _, stereo_params = self.cameras[fr_idx]
            factor = gtsam.GenericStereoFactor3D(stereo_point2D, self.NoiseModel, camera_symbol, landmark_symbol, stereo_params)
            self.graph.add(factor)
        return

    def __calculate_landmark(self, track_idx: int,
                             cameras_data: Dict[int, Tuple[int, gtsam.Pose3, gtsam.Cal3_S2Stereo]]) -> gtsam.Point3:
        single_track_data = self.tracks.xs(track_idx, level=DataBase.TRACKIDX)
        last_frame_idx = single_track_data.index.max()
        _, pose, stereo_params = self.cameras[last_frame_idx]
        stereo_cameras = gtsam.StereoCamera(pose, stereo_params)
        x_l, x_r, y = single_track_data.xs(last_frame_idx)
        landmark = stereo_cameras.backproject(gtsam.StereoPoint2(x_l, x_r, y))
        return landmark


