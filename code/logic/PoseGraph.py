import gtsam
import numpy as np
from typing import List, Dict

from models.directions import Side
from models.camera import Camera
from models.gtsam_frame import GTSAMFrame
from logic.Bundle2 import Bundle2


class PoseGraph:

    def __init__(self, bundles: List[Bundle2]):
        self.keyframe_symbols: Dict[int, int] = dict()
        self._is_optimized = False
        self._initial_estimates = gtsam.Values()
        self._optimized_estimates = gtsam.Values()
        self._factor_graph = gtsam.NonlinearFactorGraph()
        self._preprocess_bundles(bundles)

    @property
    def error(self):
        if self._is_optimized:
            return self._factor_graph.error(self._optimized_estimates)
        return self._factor_graph.error(self._initial_estimates)

    def optimize(self):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self._factor_graph, self._initial_estimates)
        self._optimized_estimates = optimizer.optimize()
        self._is_optimized = True

    def _preprocess_bundles(self, bundles: List[Bundle2]):
        # insert pose of keyframe0 to the graph
        angle_sigma, location_sigma = 0.1 * np.pi / 180, 0.1
        kf0_symbol = bundles[0].frame_symbols[0]
        kf0_pose = bundles[0].optimized_estimates.atPose3(kf0_symbol)
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([angle_sigma, angle_sigma, angle_sigma,
                                                                location_sigma, location_sigma, location_sigma]))
        prior_factor = gtsam.PriorFactorPose3(kf0_symbol, kf0_pose, pose_noise)
        self.keyframe_symbols[0] = kf0_symbol
        self._initial_estimates.insert(kf0_symbol, kf0_pose)
        self._factor_graph.add(prior_factor)

        prev_cam = Camera.from_pose3(0, kf0_pose)
        for b in bundles:
            prev_cam = self.__add_keyframe_initial_pose(b, prev_cam)
            self.__add_between_keyframe_factor(b)

    def __add_keyframe_initial_pose(self, b: Bundle2, prev_cam: Camera) -> Camera:
        """
        Calculate the global coordinates of the last Frame in the Bundle (keyframe), and add those as an initial
            estimate to the Pose Graph.
        Returns the (left) Camera associated with this keyframe, in global coordinates (required for processing the
            next Bundle)
        """
        # extract the keyframe's relative pose (relative to bundle's first frame)
        start_frame_idx, end_frame_idx = min(b.frame_symbols.keys()), max(b.frame_symbols.keys())
        start_kf_symbol, end_kf_symbol = b.frame_symbols[start_frame_idx], b.frame_symbols[end_frame_idx]
        start_kf_pose = b.optimized_estimates.atPose3(start_kf_symbol)
        end_kf_pose = b.optimized_estimates.atPose3(end_kf_symbol)
        between_pose = start_kf_pose.between(end_kf_pose)
        kf_relative_camera = Camera.from_pose3(end_frame_idx, between_pose)

        # calculate the pose in global (kf0) coordinates:
        prev_R, prev_t = prev_cam.get_rotation_matrix(), prev_cam.get_translation_vector()
        rel_R, rel_t = kf_relative_camera.get_rotation_matrix(), kf_relative_camera.get_translation_vector()
        R = rel_R @ prev_R
        t = rel_t + rel_R @ prev_t
        end_kf_global_camera = Camera(end_frame_idx, Side.LEFT, np.hstack([R, t]))

        # add the camera's global pose to the initial estimate
        end_kf_global_gtsam_frame = GTSAMFrame.from_camera(end_kf_global_camera)
        self._initial_estimates.insert(end_kf_symbol, end_kf_global_gtsam_frame.pose)
        self.keyframe_symbols[end_frame_idx] = end_kf_symbol
        return end_kf_global_camera

    def __add_between_keyframe_factor(self, b: Bundle2):
        # extract the start & end keyframes' symbols from the bundle
        start_kf_idx, end_kf_idx = min(b.frame_symbols.keys()), max(b.frame_symbols.keys())
        start_kf_bundle_symbol = b.frame_symbols[start_kf_idx]
        end_kf_bundle_symbol = b.frame_symbols[end_kf_idx]
        keys = gtsam.KeyVector()
        keys.append(start_kf_bundle_symbol)
        keys.append(end_kf_bundle_symbol)

        # calculate the relative covariance matrix
        marginals = gtsam.Marginals(b.graph, b.optimized_estimates)
        marginal_cov = marginals.jointMarginalCovariance(keys).fullMatrix()
        information = np.linalg.inv(marginal_cov)
        relative_cov = np.linalg.inv(information[-6:, -6:])

        # create and add betweenFactor to the Pose Graph
        relative_pose = b.optimized_estimates.atPose3(end_kf_bundle_symbol)
        start_kf_posegraph_symbol = self.keyframe_symbols[start_kf_idx]
        end_kf_posegraph_symbol = self.keyframe_symbols[end_kf_idx]
        noise_model = gtsam.noiseModel.Gaussian.Covariance(relative_cov)
        factor = gtsam.BetweenFactorPose3(start_kf_posegraph_symbol, end_kf_posegraph_symbol,
                                          relative_pose, noise_model)
        self._factor_graph.add(factor)


