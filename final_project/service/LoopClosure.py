import time
import numpy as np
from typing import List, Tuple

import pandas as pd

import final_project.config as c
from final_project.models.Camera import Camera
from final_project.models.Matcher import Matcher
from final_project.models.Frame import Frame
from final_project.logic.Ransac import RANSAC
from final_project.logic.PoseGraph import PoseGraph

MaxLoopsCount = 1e6
KeyframeDistanceThreshold = 10
MahalanobisThreshold = 2.0
MatchCountThreshold = 100
OutlierPercentThreshold = 20.0
LeftCam0, RightCam0 = Camera.read_initial_cameras()

# for some reason, loop matching works poorly with BF matcher (the default we use), but works
# fast and well with these parameters
_Loop_Matcher = Matcher(detector_type=c.DEFAULT_DETECTOR_NAME, matcher_type="flann", use_crosscheck=False, use_2nn=True)


def close_loops(pg: PoseGraph, max_loops_count: int = MaxLoopsCount, verbose=False) -> Tuple[List[Camera], pd.DataFrame]:
    start_time, minutes_counter = time.time(), 0
    pg.optimize()
    closed_loop_count = 0
    kf_idxs = sorted(pg.keyframe_indices)

    loop_results = []
    for i, front_idx in enumerate(kf_idxs):
        if closed_loop_count >= max_loops_count:
            break
        for j in range(i - KeyframeDistanceThreshold):
            if closed_loop_count >= max_loops_count:
                break
            curr_minute = int((time.time() - start_time) / 60)
            if verbose and curr_minute > minutes_counter:
                minutes_counter = curr_minute
                print(f"\tElapsed Minutes:\t{minutes_counter}\n\tCurrent Keyframe ID:\t{front_idx}\n")

            back_idx = kf_idxs[j]
            mahal_dist = pg.calculate_mahalanobis_distance(back_idx, front_idx)
            if mahal_dist < 0 or mahal_dist > MahalanobisThreshold:
                # these KeyFrames are not possible loops
                continue
            back_frame, front_frame, matched_indices, supporter_indices = _match_possible_loop(front_idx, back_idx)
            outlier_percent = 100 * (len(matched_indices) - len(supporter_indices)) / len(matched_indices)
            if outlier_percent > OutlierPercentThreshold:
                # there are not enough supporters to justify loop on this pair
                continue

            # reached here if this is a valid loop
            # Add edge to AdjacencyGraph & constraint to FactorGraph, then optimize
            # TODO: instead of optimizing on each loop, optimize every 5 KFs / 5 loops
            err_diff = pg.add_loop_and_optimize(back_frame, front_frame, matched_indices, supporter_indices)
            closed_loop_count += 1
            loop_results.append((front_idx, back_idx, outlier_percent, err_diff))
            if verbose:
                print(f"Loop #{closed_loop_count}")
                print(f"\tFrame{front_idx}\t<-->\tFrame{back_idx}")
                print(f"\tOutlier Percent:\t{outlier_percent:.2f}%")
                print(f"\tError Difference:\t{err_diff:.2f}")

    # one last optimization just to be sure
    pg.optimize()
    df = pd.DataFrame(loop_results, columns=[c.FrontFrame, c.BackFrame, c.OutlierPercent, c.ErrorDiff])
    return pg.extract_cameras(), df


def _match_possible_loop(front_idx: int, back_idx: int) -> Tuple[Frame, Frame, List[Tuple[int, int]], np.ndarray]:
    # Performs RANSAC on both keyframes to determine how many supporters are there for them to be at the same place
    back_frame = Frame(idx=back_idx, left_cam=LeftCam0, matcher=_Loop_Matcher)
    front_frame = Frame(idx=front_idx, matcher=_Loop_Matcher)
    matched_indices = _Loop_Matcher.match_descriptors(back_frame.descriptors, front_frame.descriptors)
    if len(matched_indices) < MatchCountThreshold:
        # not enough matches between candidates - exit early
        return back_frame, front_frame, matched_indices, np.array([])
    r = RANSAC.from_frames(back_frame, front_frame, matched_indices)
    supporter_indices, fl_cam = r.run()
    front_frame.left_cam = fl_cam
    return back_frame, front_frame, matched_indices, supporter_indices
