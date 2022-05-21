import math
import time
import gtsam
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from models.gtsam_frame import GTSAMFrame
from models.database import DataBase
from models.camera import Camera
from logic.gtsam_bundle import Bundle


class TrajectoryOptimizer:

    def __init__(self, tracks: pd.DataFrame, cameras: pd.DataFrame):
        # TODO: verify class Camera has class members initiated (K, right_rot, right_trans)
        self.tracks: pd.DataFrame = tracks
        self.gtsam_frames: pd.Series = cameras["Cam_Left"].apply(lambda cam: GTSAMFrame.from_camera(cam))
        self.landmark_symbols: pd.Series = self._extract_landmark_symbols(self.tracks)
        self.keyframe_indices: List[int] = self._choose_keyframe_indices(self.gtsam_frames.index.to_list())
        self.total_optimized = 0

    def optimize(self, verbose=False) -> Dict[int, gtsam.Values]:
        start_time, minutes_counter = time.time(), 0
        num_keyframes = len(self.keyframe_indices)
        if verbose:
            print(f"Starting trajectory optimization for {num_keyframes - 1} Bundles...\n")

        results: Dict[int, gtsam.Values] = dict()
        for i in range(num_keyframes - 1):
            start_idx, end_idx = self.keyframe_indices[i], self.keyframe_indices[i+1]

            # optimize over a single Bundle:
            bundle_frames = self.gtsam_frames[self.gtsam_frames.index.isin(np.arange(start_idx, end_idx+1))]
            bundle_track_idxs = self.tracks.index.get_level_values(level=DataBase.FRAMEIDX).isin(np.arange(start_idx, end_idx+1))
            bundle_tracks = self.tracks[bundle_track_idxs]
            bundle_landmarks = self.landmark_symbols.loc[bundle_tracks.index.unique(level=DataBase.TRACKIDX)]
            bundle = Bundle(gtsam_frames=bundle_frames, tracks_data=bundle_tracks, landmark_symbols=bundle_landmarks)
            initial_error = bundle.error
            results[i] = bundle.adjust()
            final_error = bundle.error
            self.total_optimized += initial_error - final_error

            # update pose for last Frame in this Bundle:
            # need to replace instead of update, because GTSAMFrame (NamedTuple) objects are immutable
            # see https://stackoverflow.com/questions/22562425/attributeerror-cant-set-attribute-in-python
            last_frame = self.gtsam_frames.xs(bundle_frames.index.max())
            last_frame_replacement = GTSAMFrame(symbol=last_frame.symbol,
                                                stereo_params=last_frame.stereo_params,
                                                pose=results[i].atPose3(last_frame.symbol))
            self.gtsam_frames[bundle_frames.index.max()] = last_frame_replacement

            # print every minute if verbose:
            curr_minute = int((time.time() - start_time) / 60)
            if verbose and curr_minute > minutes_counter:
                minutes_counter = curr_minute
                print(f"\tElapsed Minutes:\t{minutes_counter}\n\tIteration Number:\t{i}\n")

        elapsed = time.time() - start_time
        if verbose:
            total_minutes = elapsed / 60
            print(f"Finished Bundle adjustment within {total_minutes:.2f} minutes")
            print(f"Mean error reduction per Bundle:\t{(self.total_optimized / (num_keyframes - 1)):.3f}")
        return results

    def extract_cameras(self, optimization_results: Dict[int, gtsam.Values]) -> List[Camera]:
        cameras: List[Camera] = []
        for fr_idx in self.gtsam_frames.index:
            pose = self._extract_pose3(frame_idx=fr_idx, optimization_results=optimization_results)
            cameras.append(Camera.from_pose3(idx=fr_idx, pose=pose))
        return cameras

    def extract_landmarks(self, optimization_results: Dict[int, gtsam.Values]) -> np.ndarray:
        """
        Returns a np.ndarray of shape (3, N) where cols represent a different landmark, and rows are X,Y,Z coord.
        """
        num_landmarks = len(self.landmark_symbols.index)
        coordinates = np.zeros((num_landmarks, 3))
        for i in range(num_landmarks):
            track_idx = self.landmark_symbols.index[i]
            point_3d = self._extract_point3(track_idx, optimization_results)
            coordinates[i] = point_3d
        return coordinates.T

    def _extract_pose3(self, frame_idx: int, optimization_results: Dict[int, gtsam.Values]) -> gtsam.Pose3:
        """
        Returns a gtsam.Pose3 object containing the pose of a (left) Frame-Camera within the $optimization_results
        """
        bundle_idx = self.__calculate_bundle_for_frame(frame_idx)
        bundle_values = optimization_results[bundle_idx]
        frame_symbol = self.gtsam_frames.xs(frame_idx).symbol
        return bundle_values.atPose3(frame_symbol)

    def _extract_point3(self, track_idx: int, optimization_results: Dict[int, gtsam.Values]) -> gtsam.Point3:
        """
        Returns a gtsam.Point3 object containing the 3D coordinates of a landmark associated with the provided Track ID,
        out of all $optimization_results
        """
        last_frame_idx = self.tracks.xs(track_idx, level=DataBase.TRACKIDX).index.max()
        bundle_idx = self.__calculate_bundle_for_frame(last_frame_idx)
        bundle_values = optimization_results[bundle_idx]
        landmark_symbol = self.landmark_symbols.xs(track_idx)
        return bundle_values.atPoint3(landmark_symbol)

    def __calculate_bundle_for_frame(self, frame_idx: int) -> int:
        if frame_idx >= max(self.keyframe_indices):
            return len(self.keyframe_indices) - 2
        return list(map(lambda idx: idx > frame_idx, self.keyframe_indices)).index(True) - 1

    @staticmethod
    def _extract_landmark_symbols(tracks: pd.DataFrame) -> pd.Series:
        # matches a gtsam.symbol to each Track in the provided $tracks DF
        tracks_idxs = tracks.index.unique(level=DataBase.TRACKIDX)
        tracks_idxs_series = pd.Series(tracks_idxs)
        tracks_idxs_series.index = tracks_idxs
        return tracks_idxs_series.apply(lambda tr_idx: gtsam.symbol('l', tr_idx))

    @staticmethod
    def _choose_keyframe_indices(idxs: List[int], bundle_size: int = 15, repeating_elements: int = 1) -> List[int]:
        """
        Returns a list of integers representing the keypoint indices, where each Bundle is of size $bundle_size
        see: https://stackoverflow.com/questions/72292581/split-list-into-chunks-with-repeats-between-chunks
        """
        # TODO: enable other methods to choose keyframes (e.g. #Tracks, distance travelled, etc.)
        interval = bundle_size - repeating_elements
        bundle_idxs = [idxs[i * interval : i*interval + bundle_size] for i in range(math.ceil((len(idxs) - repeating_elements) / interval))]
        # return bundle_idxs if we want to return a list-of-list containing all indices in each bundle
        keypoint_idxs = [bundle[0] for bundle in bundle_idxs]
        if keypoint_idxs[-1] == idxs[-1]:
            return keypoint_idxs
        keypoint_idxs.append(idxs[-1])
        return keypoint_idxs


