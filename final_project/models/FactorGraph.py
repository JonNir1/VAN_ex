import gtsam
import numpy as np


class FactorGraph:

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

    def add_factor(self, factor):
        self._factor_graph.add(factor)

    def optimize(self):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self._factor_graph, self._initial_estimates)
        self._optimized_results = optimizer.optimize()
        self._is_optimized = True

    def get_optimized_pose(self, symbol: gtsam.Symbol) -> gtsam.Pose3:
        if not self.is_optimized:
            raise RuntimeError("Cannot extract pose for non-optimized FactorGraph")
        return self._optimized_results.atPose3(symbol)

    def calculate_relative_covariance(self, symbol1: gtsam.Symbol, symbol2: gtsam.Symbol) -> np.ndarray:
        keys = gtsam.KeyVector()
        keys.append(symbol1)
        keys.append(symbol2)
        marginals = gtsam.Marginals(self._factor_graph, self._optimized_results)
        marginal_cov = marginals.jointMarginalCovariance(keys).fullMatrix()
        information = np.linalg.inv(marginal_cov)
        relative_cov = np.linalg.inv(information[-6:, -6:])
        return relative_cov

    def get_pre_optimization_error(self) -> float:
        return self._factor_graph.error(self._initial_estimates)

    def get_post_optimization_error(self) -> float:
        return self._factor_graph.error(self._optimized_results)

