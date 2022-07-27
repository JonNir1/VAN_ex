import time
import gtsam
import pandas as pd
import numpy as np
from typing import List

import final_project.config as c
import final_project.logic.Utils as u
import final_project.logic.GtsamUtils as gu
from final_project.logic.Bundle import Bundle
from final_project.models.Camera import Camera


class BundleAdjustment:

    def __init__(self, tracks: pd.DataFrame, cameras: pd.Series):
        self._tracks = self.__preprocess_tracks(tracks)
        self._cameras = self.__preprocess_cameras(cameras)
        self._keyframe_indices = u.choose_keyframe_indices(max_frame_idx=tracks.index.unique(level=c.FrameIdx).max(), bundle_size=c.BUNDLE_SIZE)
        self._bundles: List[Bundle] = []
        self._reduced_error = 0.0

    def get_keyframe_indices(self) -> List[int]:
        return self._keyframe_indices

    def optimize(self, verbose=False) -> List[Camera]:
        start_time, minutes_counter = time.time(), 0
        num_bundles = len(self._keyframe_indices) - 1
        if verbose:
            print(f"Starting trajectory optimization for {num_bundles} Bundles...\n")

        for i in range(num_bundles):
            b = self._build_bundle(bundle_id=i)
            error_reduction = b.adjust()
            self._reduced_error += error_reduction
            self._bundles.append(b)

            # print every minute if verbose:
            curr_minute = int((time.time() - start_time) / 60)
            if verbose and curr_minute > minutes_counter:
                minutes_counter = curr_minute
                print(f"\tElapsed Minutes:\t{minutes_counter}\n\tIteration Number:\t{i}\n")

        relative_cameras = self._extract_cameras()
        elapsed = time.time() - start_time
        if verbose:
            total_minutes = elapsed / 60
            print(f"Finished Bundle adjustment within {total_minutes:.2f} minutes")
            print(f"Mean error reduction per Bundle:\t{(self._reduced_error / num_bundles):.3f}")
        return relative_cameras

    def extract_relative_covariances(self) -> List[np.ndarray]:
        return [b.extract_keyframes_relative_covariance() for b in self._bundles]

    def _build_bundle(self, bundle_id: int) -> Bundle:
        # Extract data for the i-th Bundle and create the Bundle object
        start_idx, end_idx = self._keyframe_indices[bundle_id], self._keyframe_indices[bundle_id + 1]
        frame_idxs = np.arange(start_idx, end_idx + 1)
        cams = self._cameras[self._cameras.index.isin(frame_idxs)]
        tracks = self._tracks[self._tracks.index.get_level_values(level=c.FrameIdx).isin(frame_idxs)]
        bundle = Bundle(cameras=cams, tracks=tracks)
        return bundle

    def _extract_cameras(self) -> List[Camera]:
        """
        Returns a list of Camera objects, after optimization of all Bundles.
        Note each Camera is aligned based on the previous one, and NOT based on the global coordinates
        """
        cameras = []
        for i, bundle in enumerate(self._bundles):
            rel_cams = bundle.extract_relative_cameras()
            if i != 0:
                # ignore the first camera in each bundle (except the first bundle)
                rel_cams = rel_cams[1:]
            for cam in rel_cams:
                cameras.append(cam)
        return cameras

    # def _extract_cameras(self) -> List[Camera]:
    #     """
    #     Returns a list of Camera objects, after optimization of all Bundles.
    #     Note each Camera is aligned based on the previous one, and NOT based on the global coordinates
    #     """
    #     first_pose = self._bundles[0].start_pose
    #     first_camera = u.calculate_camera_from_gtsam_pose(first_pose)
    #     absolute_cameras = [first_camera]
    #     for i, b in enumerate(self._bundles):
    #         kf_abs_cam = absolute_cameras.pop()  # keyframe's camera from previous Bundle
    #         kf_R, kf_t = kf_abs_cam.R, kf_abs_cam.t
    #         start_pose = b.start_pose
    #         for p in b.get_poses():
    #             between_pose = start_pose.between(p)
    #             bundle_cam = u.calculate_camera_from_gtsam_pose(between_pose)
    #             new_R = bundle_cam.R @ kf_R
    #             new_t = bundle_cam.R @ kf_t + bundle_cam.t
    #             absolute_cameras.append(Camera.from_Rt(new_R, new_t))
    #     return u.convert_to_relative_cameras(absolute_cameras)

    @staticmethod
    def __preprocess_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
        track_symbols = tracks.index.get_level_values(c.TrackIdx).map(lambda idx: gtsam.symbol('l', idx))
        return tracks.assign(Symbol=track_symbols)

    @staticmethod
    def __preprocess_cameras(cameras: pd.Series) -> pd.DataFrame:
        camera_symbols = cameras.index.map(lambda idx: gtsam.symbol('c', idx))
        camera_symbols = camera_symbols.to_series(index=cameras.index, name=c.Symbol)
        abs_cameras = u.convert_to_absolute_cameras(cameras)
        abs_poses = pd.Series([gu.calculate_gtsam_pose(abs_cam) for abs_cam in abs_cameras], name=c.AbsolutePose)
        cameras_df = pd.concat([cameras, camera_symbols, abs_poses], axis=1)
        return cameras_df



