import gtsam
import numpy as np
from typing import List, Tuple

import final_project.config as c
import final_project.logic.Utils as u
from final_project.models.Camera import Camera
from final_project.models.FactorGraph import FactorGraph
from final_project.models.AdjacencyGraph import AdjacencyGraph
from final_project.models.Frame import Frame
from final_project.models.DataBase import DataBase
from final_project.logic.Ransac import RANSAC
from final_project.logic.Bundle import Bundle


class PoseGraph:
    __MaxLoopsCount = 1e6
    _LeftCam0, _RightCam0 = Camera.read_initial_cameras()
    KeyframeDistanceThreshold = 10
    MahalanobisThreshold = 2.0
    MatchCountThreshold = 100
    OutlierPercentThreshold = 20.0

    def __init__(self, keyframes: List[int], relative_cameras: List[Camera], relative_covariances: List[np.ndarray]):
        assert len(keyframes) == len(relative_covariances) + 1, f"Keyframe count ({len(keyframes)}) should be 1 + Covariance count ({len(relative_covariances)})"
        self._keyframe_symbols = {kf: gtsam.symbol('k', kf) for kf in keyframes}
        self._factor_graph = FactorGraph()
        self._adjacency_graph = AdjacencyGraph()
        self.__build_pose_graph(relative_cameras, relative_covariances)

    @property
    def is_optimized(self) -> bool:
        return self._factor_graph.is_optimized

    def optimize(self, max_loops: int = __MaxLoopsCount, verbose=False) -> List[Camera]:
        # TODO: move this to a separate service file
        self._factor_graph.optimize()
        closed_loop_count = 0
        kf_idxs = sorted(self._keyframe_symbols.keys())
        for i, front_idx in enumerate(kf_idxs):
            if closed_loop_count >= max_loops:
                break
            for j in range(i - PoseGraph.KeyframeDistanceThreshold):
                if closed_loop_count >= max_loops:
                    break
                back_idx = kf_idxs[j]
                mahal_dist = self._calculate_mahalanobis_distance(back_idx, front_idx)
                if mahal_dist < 0 or mahal_dist > PoseGraph.MahalanobisThreshold:
                    # these KeyFrames are not possible loops
                    continue
                back_frame, front_frame, matched_indices, supporter_indices = self._match_possible_loop(front_idx, back_idx)
                outlier_percent = 100 * (len(matched_indices) - len(supporter_indices)) / len(matched_indices)
                if outlier_percent > PoseGraph.OutlierPercentThreshold:
                    # there are not enough supporters to justify loop on this pair
                    continue

                # reached here if this is a valid loop
                # Add constraint to FactorGraph and optimize
                if verbose:
                    print(f"Loop #{closed_loop_count + 1}")
                    print(f"\tFrame{front_idx}\t<-->\tFrame{back_idx}")
                back_symbol, front_symbol = self._keyframe_symbols[back_idx], self._keyframe_symbols[front_idx]
                pose, cov = self._calculate_relative_pose_and_cov(back_frame, front_frame, matched_indices, supporter_indices)
                self._factor_graph.add_between_pose_factor(back_symbol, front_symbol, pose, cov)
                self._factor_graph.optimize()  # TODO: instead of optimizing on each loop, optimize every 5 KFs / 5 loops
                closed_loop_count += 1

                # Add edge to AdjacencyGraph
                back_vertex_id = self._adjacency_graph.get_vertex_id(frame_idx=back_idx, symbol=back_symbol)
                front_vertex_id = self._adjacency_graph.get_vertex_id(frame_idx=front_idx, symbol=front_symbol)
                self._adjacency_graph.create_or_update_edge(v1_id=back_vertex_id, v2_id=front_vertex_id, cov=cov)

                if verbose:
                    prev_err = self._factor_graph.get_pre_optimization_error()
                    post_err = self._factor_graph.get_post_optimization_error()
                    print(f"\tOutlier Percent:\t{outlier_percent:.2f}%")
                    print(f"\tError Before:\t{prev_err:.4f}\n")
                    print(f"\tError After:\t{post_err:.4f}\n")

        # one last optimization just to be sure
        self._factor_graph.optimize()
        return self._extract_cameras()

    def _calculate_mahalanobis_distance(self, back_idx, front_idx) -> float:
        back_symbol = self._keyframe_symbols[back_idx]
        back_vertex_id = self._adjacency_graph.get_vertex_id(frame_idx=back_idx, symbol=back_symbol)
        front_symbol = self._keyframe_symbols[front_idx]
        front_vertex_id = self._adjacency_graph.get_vertex_id(frame_idx=front_idx, symbol=front_symbol)
        cov = self._adjacency_graph.get_relative_covariance(back_vertex_id, front_vertex_id)
        delta_cam = self._factor_graph.extract_relative_camera_matrix(front_symbol, back_symbol)
        return delta_cam @ cov @ delta_cam

    @staticmethod
    def _match_possible_loop(front_idx: int, back_idx: int):
        # Performs RANSAC on both keyframes to determine how many supporters are there for them to be at the same place
        back_frame = Frame(idx=back_idx, left_cam=PoseGraph._LeftCam0)
        front_frame = Frame(idx=front_idx)
        matched_indices = c.DEFAULT_MATCHER.match_descriptors(back_frame.descriptors, front_frame.descriptors)
        if len(matched_indices) < PoseGraph.MatchCountThreshold:
            # not enough matches between candidates - exit early
            return back_frame, front_frame, matched_indices, []
        r = RANSAC.from_frames(back_frame, front_frame, matched_indices)
        supporter_indices, fl_cam = r.run()
        front_frame.left_cam = fl_cam
        return back_frame, front_frame, matched_indices, supporter_indices

    @staticmethod
    def _calculate_relative_pose_and_cov(bf: Frame, ff: Frame,
                                         matched_indices: List[Tuple[int, int]],
                                         supporter_indices: np.ndarray) -> Tuple[gtsam.Pose3, np.ndarray]:
        # build the Bundle
        track_id = 0
        for idx in supporter_indices:
            back_idx, front_idx = matched_indices[idx]
            bf.set_track_id(back_idx, track_id)
            ff.set_track_id(front_idx, track_id)
            track_id += 1
        db = DataBase([bf, ff])
        b = Bundle(db._tracks_db, db._cameras_db)  # TODO: fix protected members
        # compute pose and covariance
        b.adjust()
        relative_pose = b.extract_keyframes_relative_pose()  # relative to $back_frame
        relative_cov = b.extract_keyframes_relative_covariance()
        return relative_pose, relative_cov

    def _extract_cameras(self) -> List[Camera]:
        absolute_cameras = []
        for kf_idx in sorted(self._keyframe_symbols.keys()):
            kf_symbol = self._keyframe_symbols[kf_idx]
            pose = self._factor_graph.get_optimized_pose(kf_symbol)
            abs_cam = u.calculate_camera_from_gtsam_pose(pose)
            absolute_cameras.append(abs_cam)
        return u.convert_to_relative_cameras(absolute_cameras)

    def __build_pose_graph(self, relative_cameras: List[Camera], relative_covariances: List[np.ndarray]):
        absolute_cameras = u.convert_to_absolute_cameras(relative_cameras)
        for i, (kf_idx, kf_symbol) in enumerate(self._keyframe_symbols.items()):
            abs_cam = absolute_cameras[kf_idx]
            abs_pose = u.calculate_gtsam_pose(abs_cam)
            self._factor_graph.add_initial_estimate(kf_symbol, abs_pose)
            v = self._adjacency_graph.create_or_update_vertex(frame_idx=kf_idx, symbol=kf_symbol)
            assert v is not None
            if i == 0:
                # this is the first Frame -> add PriorFactor
                assert kf_idx == 0
                self._factor_graph.add_prior_pose_factor(kf_symbol, abs_pose)
                prev_symbol = kf_symbol
                prev_pose = abs_pose
                continue
            # reach here if this is NOT the first Frame -> add BetweenFactor + covariance-edge
            cov = relative_covariances[i-1]
            rel_pose = prev_pose.between(abs_pose)
            self._factor_graph.add_between_pose_factor(back_symbol=prev_symbol, front_symbol=kf_symbol,
                                                       relative_pose=rel_pose, relative_covariance=cov)
            self._adjacency_graph.create_or_update_edge(v1_id=v.index - 1, v2_id=v.index, cov=cov)
            prev_symbol = kf_symbol
            prev_pose = abs_pose


