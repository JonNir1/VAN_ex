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

        prev_symbol = kf0_symbol
        prev_cam = Camera.from_pose3(0, kf0_pose)
        for b in bundles:
            keyframe_idx, keyframe_symbol = b.end_frame_index, b.end_frame_symbol
            self.keyframe_symbols[keyframe_idx] = keyframe_symbol

            # calculate keyframe's pose in the global (kf0) coordinates,and add as initial estimate:
            relative_pose = b.calculate_keyframes_relative_pose()  # relative to bundle's first frame
            global_camera = self.__calculate_global_camera(prev_cam, relative_pose)
            global_gtsam_frame = GTSAMFrame.from_camera(global_camera)
            self._initial_estimates.insert(keyframe_symbol, global_gtsam_frame.pose)

            # add relative movement as a constraint to the factor graph:
            relative_cov = b.calculate_keyframes_relative_covariance()
            noise_model = gtsam.noiseModel.Gaussian.Covariance(relative_cov)
            factor = gtsam.BetweenFactorPose3(prev_symbol, keyframe_symbol, relative_pose, noise_model)
            self._factor_graph.add(factor)

            # update values for next iteration:
            prev_symbol = keyframe_symbol
            prev_cam = global_camera
        return None

    @staticmethod
    def __calculate_global_camera(prev_cam: Camera, relative_pose: gtsam.Pose3) -> Camera:
        relative_camera = Camera.from_pose3(0, relative_pose)
        rel_R = relative_camera.get_rotation_matrix()
        rel_t = relative_camera.get_translation_vector()
        prev_R, prev_t = prev_cam.get_rotation_matrix(), prev_cam.get_translation_vector()
        R = rel_R @ prev_R
        t = rel_t + rel_R @ prev_t
        global_camera = Camera(relative_camera.idx, Side.LEFT, np.hstack([R, t]))
        return global_camera



