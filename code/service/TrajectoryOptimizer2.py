import math
import time
import gtsam
import numpy as np
import pandas as pd
from typing import List

from models.database import DataBase
from models.camera import Camera
from logic.Bundle2 import Bundle2


class TrajectoryOptimizer2:

    def __init__(self, tracks: pd.DataFrame, relative_cams: pd.Series):
        self.total_optimized = 0
        self.keyframe_idxs = self.__choose_keyframe_indices(tracks.index.unique(level=DataBase.FRAMEIDX).max(), 15)
        self.tracks = tracks
        self.relative_cameras = relative_cams
        self.bundles: List[Bundle2] = []

    def optimize(self, verbose=False):
        start_time, minutes_counter = time.time(), 0
        num_keyframes = len(self.keyframe_idxs)
        if verbose:
            print(f"Starting trajectory optimization for {num_keyframes - 1} Bundles...\n")

        for i in range(num_keyframes - 1):
            # extract data for this specific bundle:
            start_idx, end_idx = self.keyframe_idxs[i], self.keyframe_idxs[i + 1]
            bundle_frame_idxs = np.arange(start_idx, end_idx+1)
            bundle_cams = self.relative_cameras[self.relative_cameras.index.isin(bundle_frame_idxs)]
            bundle_tracks = self.tracks[self.tracks.index.get_level_values(level=DataBase.FRAMEIDX).isin(bundle_frame_idxs)]

            # optimize current bundle:
            bundle = Bundle2(bundle_cams, bundle_tracks)
            initial_error = bundle.calculate_error(bundle.initial_estimates)
            bundle.adjust()
            final_error = bundle.calculate_error(bundle.optimized_estimates)
            self.total_optimized += initial_error - final_error
            self.bundles.append(bundle)

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

    def extract_all_relative_cameras(self) -> List[Camera]:
        cameras = []
        for i, bundle in enumerate(self.bundles):
            relative_cameras = bundle.extract_all_relative_cameras()
            if i != 0:
                # ignore the first camera in each bundle (except the first bundle)
                relative_cameras = relative_cameras[1:]
            for cam in relative_cameras:
                cameras.append(cam)
        return cameras

    @staticmethod
    def __choose_keyframe_indices(max_frame_idx: int, bundle_size: int = 15):
        """
        Returns a list of integers representing the keypoint indices, where each Bundle is of size $bundle_size
        see: https://stackoverflow.com/questions/72292581/split-list-into-chunks-with-repeats-between-chunks
        """
        # TODO: enable other methods to choose keyframes (e.g. #Tracks, distance travelled, etc.)
        interval = bundle_size - 1
        all_idxs = list(range(max_frame_idx + 1))
        bundled_idxs = [all_idxs[i * interval: i * interval + bundle_size] for i in
                        range(math.ceil((len(all_idxs) - 1) / interval))]
        # return bundled_idxs # if we want to return a list-of-list containing all indices in each bundle
        keypoint_idxs = [bundle[0] for bundle in bundled_idxs]
        if keypoint_idxs[-1] == all_idxs[-1]:
            return keypoint_idxs
        keypoint_idxs.append(all_idxs[-1])
        return keypoint_idxs
