import gtsam
import numpy as np
import pandas as pd
from typing import List

from models.directions import Side
from models.camera import Camera
from models.database import DataBase
from models.gtsam_frame import GTSAMFrame


class Bundle2:
    _PixelCovariance = 1
    _LocationCovariance = 0.01
    _AngleCovariance = (0.1 * np.pi / 180) ** 2
    PointNoiseModel = gtsam.noiseModel.Isotropic.Sigma(3, _PixelCovariance)
    PoseNoiseModel = gtsam.noiseModel.Diagonal.Sigmas(np.array([_AngleCovariance, _AngleCovariance, _AngleCovariance,
                                                                _LocationCovariance, _LocationCovariance,
                                                                _LocationCovariance]))

    def __init__(self, relative_cameras: pd.Series, tracks_data: pd.DataFrame):
        self.frame_symbols = dict()
        self.landmark_symbols = dict()
        self.initial_estimates = gtsam.Values()
        self.optimized_estimates = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        self._build_bundle(relative_cameras, tracks_data)

    def adjust(self):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates)
        self.optimized_estimates = optimizer.optimize()

    def calculate_error(self, est: gtsam.Values) -> float:
        return self.graph.error(est)

    def extract_relative_cameras(self) -> List[Camera]:
        """
        Returns a list of Cameras object based on the optimized gtsam.Pose3 objects in self.optimized_values.
        The Cameras are aligned relatively to the previous Camera in the Bundle, and NOT according to the first
            keyframe of the Bundle (like the pose3 objects).
        """
        # first extract all Cameras relative to Bundle's first keyframe:
        cameras_from_kf = []
        for i, fr_idx in enumerate(self.frame_symbols.keys()):
            fr_symbol = self.frame_symbols[fr_idx]
            pose = self.optimized_estimates.atPose3(fr_symbol)
            cam_from_kf = Camera.from_pose3(idx=fr_idx, pose=pose)
            cameras_from_kf.append(cam_from_kf)

        # for each camera_from_kf, create a Camera relative to previous cam
        relative_cameras = [cameras_from_kf[0]]
        for i in range(1, len(cameras_from_kf)):
            prev_cam, curr_cam = cameras_from_kf[i-1], cameras_from_kf[i]
            prev_R, prev_t = prev_cam.get_rotation_matrix(), prev_cam.get_translation_vector()
            curr_R, curr_t = curr_cam.get_rotation_matrix(), curr_cam.get_translation_vector()
            R_rel = curr_R @ prev_R.T
            t_rel = curr_t - R_rel @ prev_t
            rel_cam = Camera(idx=curr_cam.idx, side=Side.LEFT, extrinsic_mat=np.hstack([R_rel, t_rel]))
            relative_cameras.append(rel_cam)
        return relative_cameras

    def extract_landmarks(self) -> np.ndarray:
        num_landmarks = len(self.landmark_symbols)
        landmarks = np.zeros((num_landmarks, 3))
        for i, track_symbol in enumerate(self.landmark_symbols.values()):
            landmark = self.optimized_estimates.atPoint3(track_symbol)
            landmarks[i] = landmark
        return landmarks.T

    def get_pose3(self, frame_idx: int) -> gtsam.Pose3:
        symbol = self.frame_symbols[frame_idx]
        pose = self.optimized_estimates.atPose3(symbol)
        return pose

    def get_landmark_coordinates(self, track_idx: int) -> gtsam.Point3:
        symbol = self.landmark_symbols[track_idx]
        coords = self.optimized_estimates.atPoint3(symbol)
        return coords

    def _build_bundle(self, relative_cameras: pd.Series, tracks_data: pd.DataFrame):
        frames = self.__create_and_process_frames(relative_cameras)
        self.__create_and_process_landmarks(frames, tracks_data)

    def __create_and_process_frames(self, relative_cameras: pd.Series) -> List[GTSAMFrame]:
        cameras = self.__calculate_cameras(relative_cameras)
        frames = []
        for i, cam in enumerate(cameras):
            curr_frame = GTSAMFrame.from_camera(cam)
            self.initial_estimates.insert(curr_frame.symbol, curr_frame.pose)
            self.frame_symbols[cam.idx] = curr_frame.symbol
            frames.append(curr_frame)
            if i == 0:
                # add Prior for the first frame in the Bundle
                prior_factor = gtsam.PriorFactorPose3(curr_frame.symbol, curr_frame.pose, self.PoseNoiseModel)
                self.graph.add(prior_factor)
        return frames

    def __create_and_process_landmarks(self, frames: List[GTSAMFrame], tracks_data: pd.DataFrame):
        first_frame_idx = tracks_data.index.unique(level=DataBase.FRAMEIDX).min()
        for track_idx in tracks_data.index.unique(level=DataBase.TRACKIDX):
            single_track_data = tracks_data.xs(track_idx, level=DataBase.TRACKIDX)

            # add initial est. of the landmark position
            last_frame_idx = single_track_data.index.max()
            x_l, x_r, y = single_track_data.xs(last_frame_idx)
            relative_frame_idx = last_frame_idx - first_frame_idx
            _, pose, stereo_params = frames[relative_frame_idx]
            stereo_cameras = gtsam.StereoCamera(pose, stereo_params)
            landmark_3D = stereo_cameras.backproject(gtsam.StereoPoint2(x_l, x_r, y))
            if landmark_3D[2] <= 0 or landmark_3D[2] >= 400:
                # the Z coordinate of the landmark is behind the camera or too distant
                # do not include this landmark in the bundle
                continue

            landmark_symbol = gtsam.symbol('l', track_idx)
            self.landmark_symbols[track_idx] = landmark_symbol
            self.initial_estimates.insert(landmark_symbol, landmark_3D)

            # add projection factor for each Frame participating in this Track
            for fr_idx in single_track_data.index:
                x_l, x_r, y = single_track_data.xs(fr_idx)
                stereo_point2D = gtsam.StereoPoint2(x_l, x_r, y)
                relative_frame_idx = fr_idx - first_frame_idx
                camera_symbol, _, stereo_params = frames[relative_frame_idx]
                factor = gtsam.GenericStereoFactor3D(stereo_point2D, self.PointNoiseModel, camera_symbol,
                                                     landmark_symbol, stereo_params)
                self.graph.add(factor)

    @staticmethod
    def __calculate_cameras(relative_cameras: pd.Series) -> List[Camera]:
        """ Returns a list of Camera objects aligned by the first Camera in the bundle (located at point (0,0,0)) """
        bundle_cameras = []
        for i, rel_cam in enumerate(relative_cameras):
            frame_idx = rel_cam.idx
            if i == 0:
                R, t = np.eye(3), np.zeros((3, 1))
            else:
                prev_cam = bundle_cameras[-1]
                prev_R, prev_t = prev_cam.get_rotation_matrix(), prev_cam.get_translation_vector()
                curr_R_rel, curr_t_rel = rel_cam.get_rotation_matrix(), rel_cam.get_translation_vector()
                R = curr_R_rel @ prev_R
                t = curr_t_rel + curr_R_rel @ prev_t

            cam = Camera(frame_idx, Side.LEFT, np.hstack([R, t]))
            bundle_cameras.append(cam)
        return bundle_cameras
