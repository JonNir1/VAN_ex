import time
from itertools import count

import config as c
import utils as u
from models.directions import Side, Position
from models.frame import Frame
from models.camera import Camera
from logic.ransac import Ransac


class FrameProcessor:

    def __init__(self, verbose: bool = False):
        self._verbose = verbose
        self._track_counter = count(0)

    def process_frames(self, num_frames: int = Frame.MaxIndex + 1) -> tuple[list[Frame], float]:
        start_time, minutes_counter = time.time(), 0
        max_frame_count = Frame.MaxIndex + 1
        assert 1 < num_frames <= max_frame_count, f"Must process between 2 and {max_frame_count} frames"
        if self._verbose:
            print(f"Starting to process trajectory for {num_frames} tracking-pairs...")

        all_frames = [self._process_first_frame()]
        for i in range(1, num_frames):
            back_frame = all_frames[-1]
            front_frame = Frame(i)
            self._process_frame_pair(back_frame, front_frame)
            all_frames.append(front_frame)

            # print if needed:
            curr_minute = int((time.time() - start_time) / 60)
            if self._verbose and curr_minute > minutes_counter:
                minutes_counter = curr_minute
                print(f"\tProcessed {i} tracking-pairs in {minutes_counter} minutes")

        total_elapsed = time.time() - start_time
        if self._verbose:
            total_minutes = total_elapsed / 60
            print(f"Finished running for all tracking-pairs. Total runtime: {total_minutes:.2f} minutes")
        return all_frames, total_elapsed

    def _process_frame_pair(self, back_frame: Frame, front_frame: Frame):
        # Finds Camera matrices for the front_frame, and supporting MutualMatch objects for these Cameras
        mutual_matches = c.MATCHER.match_between_frames(back_frame, front_frame)
        fl_cam, fr_cam, supporting_matches = Ransac().run(mutual_matches, back_frame.left_camera,
                                                          back_frame.right_camera, self._verbose)
        front_frame.left_camera = fl_cam
        front_frame.right_camera = fr_cam

        # update tracks:
        # TODO: fix bug in tracking logic (why do we have Tracks of length 1?)
        for supporter in supporting_matches:
            back_match = supporter.get_frame_match(Position.BACK)
            front_match = supporter.get_frame_match(Position.FRONT)
            track_id = back_frame.match_to_track_id.get(back_match)
            if track_id is None:
                # this is a new Track that needs to be added to the back_frame
                track_id = next(self._track_counter)
                back_frame.match_to_track_id[back_match] = track_id
            front_frame.match_to_track_id[front_match] = track_id

    @staticmethod
    def _process_first_frame():
        first_frame = Frame(0)
        K, M_left, M_right = u.read_first_camera_matrices()
        first_frame.left_camera = Camera(0, Side.LEFT, M_left)
        first_frame.right_camera = Camera(0, Side.RIGHT, M_right)
        return first_frame



