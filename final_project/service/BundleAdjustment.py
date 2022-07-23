import time
import math
import gtsam
import pandas as pd
import numpy as np
from typing import List

import final_project.config as c
import final_project.camera_utils as cu
from final_project.logic.Bundle import Bundle
from final_project.models.Camera import Camera


class BundleAdjustment:

    def __init__(self, tracks: pd.DataFrame, cameras: pd.Series):
        self._tracks = self.__preprocess_tracks(tracks)
        self._cameras = self.__preprocess_cameras(cameras)
        self._keyframe_indices = self.__choose_keyframe_indices(tracks.index.unique(level=c.FrameIdx).max())
        self._bundles: List[Bundle] = []
        self._reduced_error = 0.0

    def optimize(self, verbose=False):
        start_time, minutes_counter = time.time(), 0
        num_bundles = len(self._keyframe_indices) - 1
        if verbose:
            print(f"Starting trajectory optimization for {num_bundles} Bundles...\n")

        for i in range(num_bundles):
            # extract data for this specific bundle:
            start_idx, end_idx = self._keyframe_indices[i], self._keyframe_indices[i + 1]
            frame_idxs = np.arange(start_idx, end_idx + 1)
            cams = self._cameras[self._cameras.index.isin(frame_idxs)]
            tracks = self._tracks[self._tracks.index.get_level_values(level=c.FrameIdx).isin(frame_idxs)]

            # optimize current bundle:
            bundle = Bundle(cameras=cams, tracks=tracks)
            error_reduction = bundle.adjust()
            self._reduced_error += error_reduction
            self._bundles.append(bundle)

            # print every minute if verbose:
            curr_minute = int((time.time() - start_time) / 60)
            if verbose and curr_minute > minutes_counter:
                minutes_counter = curr_minute
                print(f"\tElapsed Minutes:\t{minutes_counter}\n\tIteration Number:\t{i}\n")

        elapsed = time.time() - start_time
        if verbose:
            total_minutes = elapsed / 60
            print(f"Finished Bundle adjustment within {total_minutes:.2f} minutes")
            print(f"Mean error reduction per Bundle:\t{(self._reduced_error / num_bundles):.3f}")

    def extract_cameras(self) -> List[Camera]:
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

    @staticmethod
    def __preprocess_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
        track_symbols = tracks.index.get_level_values(c.TrackIdx).map(lambda idx: gtsam.symbol('l', idx))
        return tracks.assign(Symbol=track_symbols)

    @staticmethod
    def __preprocess_cameras(cameras: pd.Series) -> pd.DataFrame:
        camera_symbols = cameras.index.map(lambda idx: gtsam.symbol('c', idx))
        camera_symbols = camera_symbols.to_series(index=cameras.index, name=c.Symbol)
        abs_cameras = cu.convert_to_absolute_cameras(cameras)
        abs_poses = pd.Series([cu.calculate_gtsam_pose(abs_cam) for abs_cam in abs_cameras], name=c.AbsolutePose)
        cameras_df = pd.concat([cameras, camera_symbols, abs_poses], axis=1)
        return cameras_df

    @staticmethod
    def __choose_keyframe_indices(max_frame_idx: int, bundle_size: int = c.BUNDLE_SIZE):
        """
        Returns a list of integers representing the keypoint indices, where each Bundle is of size $bundle_size
        see: https://stackoverflow.com/questions/72292581/split-list-into-chunks-with-repeats-between-chunks
        """
        # TODO: enable other methods to choose keyframes (e.g. #Tracks, distance travelled, etc.)
        interval = bundle_size - 1
        all_idxs = list(range(max_frame_idx + 1))
        bundled_idxs = [all_idxs[i * interval: i * interval + bundle_size] for i in
                        range(math.ceil((len(all_idxs) - 1) / interval))]
        # return bundled_idxs  # if we want to return a list-of-list containing all indices in each bundle
        keypoint_idxs = [bundle[0] for bundle in bundled_idxs]
        if keypoint_idxs[-1] == all_idxs[-1]:
            return keypoint_idxs
        keypoint_idxs.append(all_idxs[-1])
        return keypoint_idxs



