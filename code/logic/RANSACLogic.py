import time
import random
import numpy as np

from models.Camera2 import Camera2
from logic.TriangulationLogic import triangulate_pixels
from logic.PnPLogic import pnp


class RANSAC:
    MinimalSetSize = 4
    DefaultSuccessProbability = 0.9999
    MaxDistanceForSupporter = 2

    def __init__(self, success_prob: float = DefaultSuccessProbability):
        assert 0 < success_prob < 1, f"Success Probability must be between 0 and 1 (exclusive)"
        self._success_probability = success_prob
        self._outlier_probability = 0.99  # this value is updated during the ransac run
        self._num_iterations = self.__calculate_number_of_iteration()  # this value is updated during the ransac run

    def run(self, back_features: np.ndarray, front_features: np.ndarray, verbose=False):
        start_time = time.time()
        if verbose:
            print(f"Starting RANSAC with {self._num_iterations} iterations")

        initial_supporter_idxs = self._find_initial_supporters(back_features, front_features, verbose)
        fl_cam, supporter_idxs = self._refine_model(back_features, front_features, initial_supporter_idxs, verbose)

        elapsed = time.time() - start_time
        if verbose:
            print(f"Completed RANSAC in {elapsed:.2f} seconds\n\tNumber of Supporters: {len(supporter_idxs)}\n")
        return fl_cam, supporter_idxs

    def _find_initial_supporters(self, back_features: np.ndarray, front_features: np.ndarray,
                                 verbose=False) -> np.ndarray:
        """
        First loop of RANSAC: find a model with as many supporters as possible, using a minimal-set (4) of mutual features.

        In each iteration sample 4 mutual features and calculate front cameras based on them. Then, project all
            3D-points to the cameras' planes and see how many are projected within a given distance from their real
            cv2.KeyPoint pixels (default distance 2). Those points are considered supporters.

        Returns the indices of supporting Features for the best model that was found.
        """
        if verbose:
            print("RANSAC: looking for initial supporters")
        self.__verify_features_shape(back_features, front_features)
        num_features = back_features.shape[0]
        back_left_pixels, back_right_pixels = back_features[:, :2], back_features[:, 2:]
        back_landmarks_3d = triangulate_pixels(back_left_pixels, back_right_pixels)

        best_supporting_indices = np.array([])
        while self._num_iterations > 0:
            relative_fl_cam = None
            while relative_fl_cam is None:
                random_idxs = random.sample(range(num_features), self.MinimalSetSize)
                sample_back_features = back_features[random_idxs]
                sample_front_features = front_features[random_idxs]
                relative_fl_cam = pnp(sample_back_features, sample_front_features)

            assert isinstance(relative_fl_cam, Camera2)
            supporting_indices = self._estimate_model(relative_fl_cam, front_features, back_landmarks_3d)

            if len(supporting_indices) > len(best_supporting_indices):
                best_supporting_indices = supporting_indices
                num_supporters = len(best_supporting_indices)
                self._outlier_probability = 1 - num_supporters / num_features
                self._num_iterations = self.__calculate_number_of_iteration()
                if verbose:
                    print(f"\tUpdated Parameters:\tNum. Supporters: {num_supporters}" +
                          f"\tRem. Iterations: {self._num_iterations}")
            else:
                self._num_iterations -= 1
                if verbose and self._num_iterations % 100 == 0:
                    print(f"\tCurrent Parameters:\tNum. Supporters: {len(best_supporting_indices)}" +
                          f"\tRem. Iterations: {self._num_iterations}")
        return best_supporting_indices

    def _refine_model(self, back_features: np.ndarray, front_features: np.ndarray, supporting_indices: np.ndarray,
                      verbose=False):
        if verbose:
            print("RANSAC: refining best model")
        self.__verify_features_shape(back_features, front_features)
        back_left_pixels, back_right_pixels = back_features[:, :2], back_features[:, 2:]
        back_landmarks_3d = triangulate_pixels(back_left_pixels, back_right_pixels)

        while True:
            supporters_back = back_features[supporting_indices]
            supporters_front = front_features[supporting_indices]
            relative_fl_cam = pnp(supporters_back, supporters_front)
            assert isinstance(relative_fl_cam, Camera2)
            curr_supporting_indices = self._estimate_model(relative_fl_cam, front_features, back_landmarks_3d)

            if len(curr_supporting_indices) > len(supporting_indices):
                # we can refine the model even further in the next iteration
                supporting_indices = curr_supporting_indices
            else:  # this is the best model we can find, exit loop
                return relative_fl_cam, supporting_indices

    @staticmethod
    def _estimate_model(front_left_cam: Camera2, front_features: np.ndarray,
                        landmarks_3d: np.ndarray) -> np.ndarray:
        assert front_features.shape[0] == 4 or front_features.shape[1] == 4, \
            "Argument $front_features is not a 4D array"
        if front_features.shape[0] != 4:
            front_features = front_features.T

        front_left_pixels, front_right_pixels = front_features[:2], front_features[2:]
        supporting_indices_left = RANSAC._find_supporters_for_camera(front_left_cam, landmarks_3d, front_left_pixels)
        front_right_camera = front_left_cam.calculate_right_camera()
        supporting_indices_right = RANSAC._find_supporters_for_camera(front_right_camera, landmarks_3d,
                                                                      front_right_pixels)
        supporting_indices = np.intersect1d(supporting_indices_left, supporting_indices_right, return_indices=False)
        return supporting_indices

    @staticmethod
    def _find_supporters_for_camera(cam: Camera2, landmarks_3d: np.ndarray,
                                    keypoint_pixels_2d: np.ndarray) -> np.ndarray:
        assert landmarks_3d.shape[0] == 3 or landmarks_3d.shape[1] == 3, \
            "Argument $landmarks_3d is not a 3D-points array"
        if landmarks_3d.shape[0] != 3:
            landmarks_3d = landmarks_3d.T

        assert keypoint_pixels_2d.shape[0] == 2 or keypoint_pixels_2d.shape[1] == 2, \
            "Argument $actual_projections is not a 2D-points array"
        if keypoint_pixels_2d.shape[0] != 2:
            landmark_pixels_2d = keypoint_pixels_2d.T

        landmark_projections = cam.project(landmarks_3d)  # shape (2, N)
        assert keypoint_pixels_2d.shape == landmark_projections.shape
        euclidean_distances = np.linalg.norm(landmark_projections - keypoint_pixels_2d, ord=2, axis=0)
        supporting_indices = np.where(euclidean_distances <= RANSAC.MaxDistanceForSupporter)
        return supporting_indices[0]

    def __calculate_number_of_iteration(self) -> int:
        """
        Calculate how many iterations of RANSAC are required to get good enough results,
        i.e. for a set of size $s, with outlier probability $e and success probability $p
        we need N > log(1-$p) / log(1-(1-$e)^$s)

        :return: N: int -> number of iterations
        """
        nom = np.log(1 - self._success_probability)
        good_set_prob = np.power(1 - self._outlier_probability, self.MinimalSetSize)
        denominator = np.log(1 - good_set_prob)
        return int(nom / denominator) + 1

    @staticmethod
    def __verify_features_shape(back_features: np.ndarray, front_features: np.ndarray):
        assert back_features.shape[1] == 4, f"Back Features must include 4 columns"
        assert front_features.shape[1] == 4, f"Front Features must include 4 columns"
        assert back_features.shape[0] == front_features.shape[0], f"Back & Front Features count mismatch"
