import gtsam
import numpy as np


class FactorGraph:
    _PixelSigma = 1
    _LocationSigma = 0.1
    _AngleSigma = 0.1 * np.pi / 180

    PointNoiseModel = gtsam.noiseModel.Isotropic.Sigma(3, _PixelSigma)
    PoseNoiseModel = gtsam.noiseModel.Diagonal.Sigmas(np.array([_AngleSigma, _AngleSigma, _AngleSigma,
                                                                _LocationSigma, _LocationSigma, _LocationSigma]))

    def __init__(self):
        self._is_optimized = False
        self._initial_estimates = gtsam.Values()
        self._optimized_results = gtsam.Values()
        self._factor_graph = gtsam.NonlinearFactorGraph()

    @property
    def is_optimized(self) -> bool:
        return self._is_optimized

    def add_initial_estimate(self, symbol: gtsam.Symbol, value):
        self._initial_estimates.insert(symbol, value)

    def add_prior_pose_factor(self, symbol: gtsam.Symbol, pose: gtsam.Pose3):
        prior_factor = gtsam.PriorFactorPose3(symbol, pose, self.PoseNoiseModel)
        self._factor_graph.add(prior_factor)

    def add_between_pose_factor(self, back_symbol: gtsam.Symbol, front_symbol: gtsam.Symbol,
                                relative_pose: gtsam.Pose3, relative_covariance: np.ndarray):
        noise_model = gtsam.noiseModel.Gaussian.Covariance(relative_covariance)
        between_factor = gtsam.BetweenFactorPose3(back_symbol, front_symbol, relative_pose, noise_model)
        self._factor_graph.add(between_factor)

    def add_stereo_projection_factor(self, camera_symbol: gtsam.Symbol, landmark_symbol: gtsam.Symbol,
                                     stereo_point: gtsam.StereoPoint2, stereo_params: gtsam.Cal3_S2Stereo):
        stereo_factor = gtsam.GenericStereoFactor3D(stereo_point, self.PointNoiseModel, camera_symbol, landmark_symbol, stereo_params)
        self._factor_graph.add(stereo_factor)

    def optimize(self):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self._factor_graph, self._initial_estimates)
        self._optimized_results = optimizer.optimize()
        self._is_optimized = True

    def get_optimized_pose(self, symbol: gtsam.Symbol) -> gtsam.Pose3:
        if not self.is_optimized:
            raise RuntimeError("Cannot extract pose for non-optimized FactorGraph")
        return self._optimized_results.atPose3(symbol)

    def calculate_relative_covariance(self, symbol1: gtsam.Symbol, symbol2: gtsam.Symbol) -> np.ndarray:
        if not self.is_optimized:
            raise RuntimeError("Cannot calculate covariance from a non-optimized FactorGraph")
        keys = gtsam.KeyVector()
        keys.append(symbol1)
        keys.append(symbol2)
        marginals = gtsam.Marginals(self._factor_graph, self._optimized_results)
        marginal_cov = marginals.jointMarginalCovariance(keys).fullMatrix()
        information = np.linalg.inv(marginal_cov)
        relative_cov = np.linalg.inv(information[-6:, -6:])
        return relative_cov

    def extract_relative_camera_matrix(self, front_symbol: gtsam.Symbol, back_symbol: gtsam.Symbol) -> np.ndarray:
        if not self.is_optimized:
            raise RuntimeError("Cannot extract relative 6D camera matrix from a non-optimized FactorGraph")
        front_pose = self.get_optimized_pose(front_symbol)
        back_pose = self.get_optimized_pose(back_symbol)
        relative_pose = back_pose.between(front_pose)
        delta_cam = np.hstack([relative_pose.rotation().xyz(), relative_pose.translation()])
        return delta_cam

    def get_pre_optimization_error(self) -> float:
        return self._factor_graph.error(self._initial_estimates)

    def get_post_optimization_error(self) -> float:
        return self._factor_graph.error(self._optimized_results)

    def calculate_optimization_error_reduction(self) -> float:
        return self.get_pre_optimization_error() - self.get_post_optimization_error()

