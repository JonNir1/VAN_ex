import numpy as np
from typing import List, Tuple

import final_project.config as c
from final_project.models.Camera import Camera
from final_project.models.Frame import Frame
from final_project.logic.Ransac import RANSAC
from final_project.logic.PoseGraph import PoseGraph

MaxLoopsCount = 1e6
KeyframeDistanceThreshold = 10
MahalanobisThreshold = 2.0
MatchCountThreshold = 100
OutlierPercentThreshold = 20.0
LeftCam0, RightCam0 = Camera.read_initial_cameras()


def close_loops(pg: PoseGraph, max_loops_count: int = MaxLoopsCount, verbose=False):
    pg.optimize()
    closed_loop_count = 0
    kf_idxs = sorted(pg.keyframe_indices)
    for i, front_idx in enumerate(kf_idxs):
        if closed_loop_count >= max_loops_count:
            break
        for j in range(i - KeyframeDistanceThreshold):
            if closed_loop_count >= max_loops_count:
                break
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
            if verbose:
                print(f"Loop #{closed_loop_count + 1}")
                print(f"\tFrame{front_idx}\t<-->\tFrame{back_idx}")
                print(f"\tOutlier Percent:\t{outlier_percent:.2f}%")
            # Add edge to AdjacencyGraph & constraint to FactorGraph, then optimize
            # TODO: instead of optimizing on each loop, optimize every 5 KFs / 5 loops
            pg.add_loop_and_optimize(back_frame, front_frame, matched_indices, supporter_indices)
            closed_loop_count += 1

    # one last optimization just to be sure
    pg.optimize()
    return pg.extract_cameras()


def _match_possible_loop(front_idx: int, back_idx: int) -> Tuple[Frame, Frame, List[Tuple[int, int]], np.ndarray]:
    # Performs RANSAC on both keyframes to determine how many supporters are there for them to be at the same place
    back_frame = Frame(idx=back_idx, left_cam=LeftCam0)
    front_frame = Frame(idx=front_idx)
    matched_indices = c.DEFAULT_MATCHER.match_descriptors(back_frame.descriptors, front_frame.descriptors)
    if len(matched_indices) < MatchCountThreshold:
        # not enough matches between candidates - exit early
        return back_frame, front_frame, matched_indices, np.array([])
    r = RANSAC.from_frames(back_frame, front_frame, matched_indices)
    supporter_indices, fl_cam = r.run()
    front_frame.left_cam = fl_cam
    return back_frame, front_frame, matched_indices, supporter_indices
