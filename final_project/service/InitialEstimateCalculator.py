import time
from typing import List
from itertools import count

import final_project.config as c
from final_project.models.Matcher import Matcher, DEFAULT_MATCHER
from final_project.models.Camera import Camera
from final_project.models.Frame import Frame
from final_project.models.DataBase import DataBase
from final_project.logic.Ransac import RANSAC


class IECalc:

    MinTrackLength = 3

    def __init__(self, matcher: Matcher = DEFAULT_MATCHER):
        self._matcher = matcher
        self._cam0_l, self._cam0_r = Camera.read_initial_cameras()
        self._track_count = count(0)

    @property
    def num_tracks(self):
        return self._track_count

    def process(self,
                num_frames: int = c.NUM_FRAMES,
                min_track_length: int = MinTrackLength,
                verbose=False, should_save=False) -> List[Frame]:
        start_time, minutes_counter = time.time(), 0
        if verbose:
            print(f"Calculating initial estimates for {num_frames} Frames...")

        first_frame = Frame(0, matcher=self._matcher)
        first_frame.left_cam = self._cam0_l

        processed_frames = [first_frame]
        for i in range(1, num_frames):
            next_frame = self._process_next_frame(processed_frames[-1])
            processed_frames.append(next_frame)
            curr_minute = int((time.time() - start_time) / 60)
            if verbose and curr_minute > minutes_counter:
                # print updates every minute
                minutes_counter = curr_minute
                print(f"\tProcessed {i} tracking-pairs in {minutes_counter} minutes")

        if should_save:
            db = DataBase(processed_frames)
            db.prune_short_tracks(min_track_length, inplace=True)
            db.to_pickle()
        elapsed = time.time() - start_time
        if verbose:
            total_minutes = elapsed / 60
            print(f"Processed all {num_frames} Frames in {total_minutes:.2f} minutes")
        return processed_frames

    def _process_next_frame(self, bf: Frame) -> Frame:
        """
        For two consecutive Frames,
        1) Find mutual keypoints and triangulate them using both back-Frame's Cameras
        2) Use the 3D landmarks and corresponding features on the front Frame, to run RANSAC and calculate:
            a) the front Frame's left Camera
            b) a maximal subset of landmarks supporting the calculated Camera
        3) Iterate over supporting landmarks and assign a track ID to each of their projections on both Frames
            use mutual keypoints to extract the front Frame's left Camera
        4) Updates the back Frame with newly assigned tracks, and returns the (new) front Frame.

        :param bf: back frame
        :returns ff: front frame - the newly created Frame with updated tracks linking to $bf
        """

        # use RANSAC to calculate front-left Camera & supporting landmark indices
        ff = Frame(bf.idx + 1, matcher=self._matcher)
        matched_indices = self._matcher.match_descriptors(bf.descriptors, ff.descriptors)
        r = RANSAC.from_frames(bf, ff, matched_indices)
        supporting_idxs, fl_cam = r.run()

        # update Frames:
        ff.left_cam = fl_cam
        for idx in supporting_idxs:
            back_idx, front_idx = matched_indices[idx]
            track_id = bf.get_track_id(back_idx)
            if track_id is None:
                track_id = next(self._track_count)
                bf.set_track_id(back_idx, track_id)
            ff.set_track_id(front_idx, track_id)
        return ff


