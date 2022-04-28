import time

import numpy as np

from models.directions import Position
from models.keypoint import KeyPoint
from models.frame import Frame
from models.camera import Camera
from models.match import FrameMatch
from models.track import Track
from logic.keypoints_matching import detect_and_match, match_between_frames
from logic.ransac import Ransac
from logic.updater import Updater


def estimate_trajectory(num_frames: int = Frame.MaxIndex + 1, verbose=False) -> tuple[list[Frame], np.ndarray, float]:
    start_time, minutes_counter = time.time(), 0
    max_frame_count = Frame.MaxIndex + 1
    assert 1 < num_frames <= max_frame_count, f"Must process between 2 and {max_frame_count} frames"

    if verbose:
        print(f"Starting to process trajectory for {num_frames} tracking-pairs...")

    all_frames = [_process_first_frame()]
    for i in range(1, num_frames):
        back_frame = all_frames[-1]
        front_frame = Frame(i)
        mutual_matches, new_front_matches = match_between_frames(back_frame, front_frame)

        ransac = Ransac()
        fl_cam, fr_cam, supporting_matches = ransac.run(mutual_matches, back_frame.left_camera, back_frame.right_camera, verbose)

        # add the non-supporting front-matches to the list of "new" matches:
        new_front_matches.extend([m.get_frame_match(Position.FRONT)
                                  for m in mutual_matches if m not in supporting_matches])
        updater = Updater(back_frame, front_frame, supporting_matches)
        updater.update(new_front_matches, fl_cam, fr_cam)
        all_frames.append(front_frame)

        # print if needed:
        curr_minute = int((time.time() - start_time) / 60)
        if verbose and curr_minute > minutes_counter:
            minutes_counter = curr_minute
            print(f"\tProcessed {i} tracking-pairs in {minutes_counter} minutes")

    # calculate the estimated trajectory
    estimated_trajectory = _calculate_trajectory(all_frames)
    total_elapsed = time.time() - start_time
    if verbose:
        total_minutes = total_elapsed / 60
        print(f"Finished running for all tracking-pairs. Total runtime: {total_minutes:.2f} minutes")
    return all_frames, estimated_trajectory, total_elapsed


def _process_first_frame():
    first_frame = Frame(0)
    left_cam0, right_cam0 = Camera.read_first_cameras()
    first_frame.left_camera = left_cam0
    first_frame.right_camera = right_cam0
    cv2_kps_left, _, cv2_kps_right, _, cv2_inliers = detect_and_match(first_frame, cross_check=False)
    for inlier in cv2_inliers:
        kp_l = KeyPoint.from_cv2_keypoint(cv2_kps_left[inlier.queryIdx])
        kp_r = KeyPoint.from_cv2_keypoint(cv2_kps_right[inlier.trainIdx])
        match = FrameMatch(kp_l, kp_r)
        tr = Track(0)
        first_frame.add_track(tr, match)
    return first_frame


def _calculate_trajectory(frames: list[Frame]) -> np.ndarray:
    """
    Calculates the left-camera trajectory of all frames.
    Returns a 3xN array representing the 3D location of all (left-)cameras of the provided Frames
    """
    num_samples = len(frames)
    trajectory = np.zeros((num_samples, 3))
    for i, fr in enumerate(frames):
        l_cam = fr.left_camera
        if l_cam is None:
            continue
        R = l_cam.get_rotation_matrix()
        t = l_cam.get_translation_vector()
        trajectory[i] -= (R.T @ t).reshape((3,))
    return trajectory.T


