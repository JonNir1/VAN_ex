import gtsam
import numpy as np
from typing import List, Tuple

import final_project.logic.Utils as u
import final_project.logic.GtsamUtils as gu
from final_project.models.Camera import Camera
from final_project.models.FactorGraph import FactorGraph
from final_project.models.AdjacencyGraph import AdjacencyGraph
from final_project.models.Frame import Frame
from final_project.models.DataBase import DataBase
from final_project.logic.Bundle import Bundle


class PoseGraph:

    def __init__(self, keyframes: List[int], relative_cameras: List[Camera], relative_covariances: List[np.ndarray]):
        assert len(keyframes) == len(relative_covariances) + 1, f"Keyframe count ({len(keyframes)}) should be 1 + Covariance count ({len(relative_covariances)})"
        self._keyframe_symbols = {kf: gtsam.symbol('k', kf) for kf in keyframes}
        self._factor_graph = FactorGraph()
        self._adjacency_graph = AdjacencyGraph()
        self.__build_pose_graph(relative_cameras, relative_covariances)

    @property
    def is_optimized(self) -> bool:
        return self._factor_graph.is_optimized

    @property
    def keyframe_indices(self) -> List[int]:
        return sorted(self._keyframe_symbols.keys())

    def optimize(self):
        self._factor_graph.optimize()

    def calculate_mahalanobis_distance(self, back_idx, front_idx) -> float:
        back_symbol = self._keyframe_symbols[back_idx]
        back_vertex_id = self._adjacency_graph.get_vertex_id(frame_idx=back_idx, symbol=back_symbol)
        front_symbol = self._keyframe_symbols[front_idx]
        front_vertex_id = self._adjacency_graph.get_vertex_id(frame_idx=front_idx, symbol=front_symbol)
        cov = self._adjacency_graph.get_relative_covariance(back_vertex_id, front_vertex_id)
        delta_cam = self._factor_graph.extract_relative_camera_matrix(front_symbol, back_symbol)
        return delta_cam @ cov @ delta_cam

    def add_loop_and_optimize(self, bf: Frame, ff: Frame, match_idxs: List[Tuple[int, int]], supporters: np.ndarray):
        """
        The given Frames are a loop in the trajectory. This calculates their relative covariance and adds an edge to
        the AdjacencyGraph with this covariance as weight. Then, a new constraint is added to the FactorGraph and it is
        subsequently re-optimized.
        """
        back_idx, front_idx = bf.idx, ff.idx
        back_symbol, front_symbol = self._keyframe_symbols[back_idx], self._keyframe_symbols[front_idx]
        pose, cov = self._calculate_relative_pose_and_cov(bf, ff, match_idxs, supporters)

        # Add edge to AdjacencyGraph
        back_vertex_id = self._adjacency_graph.get_vertex_id(frame_idx=back_idx, symbol=back_symbol)
        front_vertex_id = self._adjacency_graph.get_vertex_id(frame_idx=front_idx, symbol=front_symbol)
        self._adjacency_graph.create_or_update_edge(v1_id=back_vertex_id, v2_id=front_vertex_id, cov=cov)

        # Add constraint to FactorGraph
        self._factor_graph.add_between_pose_factor(back_symbol, front_symbol, pose, cov)
        self.optimize()

    def extract_cameras(self) -> List[Camera]:
        # after optimization is finished, returns a list of (relative) Cameras
        # @raises RuntimeError if called before any optimization
        absolute_cameras = []
        for kf_idx in sorted(self._keyframe_symbols.keys()):
            kf_symbol = self._keyframe_symbols[kf_idx]
            pose = self._factor_graph.get_optimized_pose(kf_symbol)
            abs_cam = gu.calculate_camera_from_gtsam_pose(pose)
            absolute_cameras.append(abs_cam)
        return u.convert_to_relative_cameras(absolute_cameras)

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

    def __build_pose_graph(self, relative_cameras: List[Camera], relative_covariances: List[np.ndarray]):
        absolute_cameras = u.convert_to_absolute_cameras(relative_cameras)
        for i, (kf_idx, kf_symbol) in enumerate(self._keyframe_symbols.items()):
            abs_cam = absolute_cameras[kf_idx]
            abs_pose = gu.calculate_gtsam_pose(abs_cam)
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


