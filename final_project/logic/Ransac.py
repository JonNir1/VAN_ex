import time
import numpy as np
from typing import Tuple

from final_project.models.Camera import Camera
from final_project.logic.PnP import pnp
from final_project.logic.Projection import project


class RANSAC:

    _MinimalSetSize = 4
    _DefaultSuccessProbability = 0.9999
    _SupporterThreshold = 2
    _MaxIterationCount = int(1e6)

    def __init__(self, back_landmarks: np.ndarray, front_features: np.ndarray,
                 success_prob: float = _DefaultSuccessProbability, verbose=False):
        landmarks_num, landmarks_dims = back_landmarks.shape
        features_num, features_dims = front_features.shape
        assert landmarks_dims == 3, "landmarks should have 3 cols"
        assert features_dims == 4, "front_features should have 4 cols"
        assert landmarks_num == features_num, f"unequal number of rows for landmarks ({landmarks_num}) and pixels ({features_num})"
        assert 0 < success_prob < 1, f"Success Probability ({success_prob:.2f}) must be a fraction between 0 and 1"
        self._landmarks = back_landmarks
        self._front_features = front_features
        self._success_probability = success_prob
        self._outlier_probability = 0.99
        self._remaining_iterations = self.__calculate_remaining_iterations()
        self._performed_iterations = 0
        self._verbose = verbose

    @property
    def max_num_supporters(self) -> int:
        return self._landmarks.shape[0]

    @property
    def performed_iterations(self) -> int:
        return self._performed_iterations

    @property
    def inlier_probability(self) -> float:
        return 1 - self._outlier_probability

    def run(self) -> Tuple[np.ndarray, Camera]:
        """
        Runs the RANSAC algorithm (two loops) to calculate the best front-left Camera and supporting landmarks.
        :return:
            - supporter_indices: np array of indices, representing the supporting landmarks for the best Camera
            - fl_cam: Camera; the front-left Camera calculated using PnP on the set of supporters
        """
        start = time.time()
        if self._verbose:
            print("\tRANSAC:\tinitial estimation loop")
        supporter_indices = self._estimate_model()
        elapsed = time.time() - start
        iters = self.performed_iterations
        if self._verbose:
            print(f"\t\tTime:\t{elapsed:.2f}\tIterations:\t{iters}")
            print("\tRANSAC:\trefinement loop")
        supporter_indices, fl_cam = self._refine_model(supporter_indices)
        elapsed = time.time() - start
        iters = self.performed_iterations - iters
        if self._verbose:
            print(f"\t\tTime:\t{elapsed:.2f}\tIterations:\t{iters}")
        return supporter_indices, fl_cam

    def _estimate_model(self) -> np.ndarray:
        """
        First loop of RANSAC algorithm:
        1) Choose a subset of {4} indices from the whole list of available 3D landmarks.
        2) Given these indices, extract the corresponding landmarks triangulated using the back-Camera, and
            corresponding projections of these landmarks onto the front-left Camera.
        3) Use the landmarks & pixels to calculate the front-left Camera using PnP, and then calculate the front-right
            Camera using the Cameras left-to-right transformation.
        4. Project the whole set of landmarks onto both front Cameras, and calculate the Euclidean distance of projected
            pixels from the actual keypoints' pixels. Landmarks with distance lower than {2} on both cameras are
            considered supporters for these Cameras.
        5. Use the number of supporters to re-calculate the number of required iterations, and start again.
        6. Finally, return the list of best-supporting indices.
        """
        best_supporting_indices = np.array([])
        while self._remaining_iterations > 0:
            self._performed_iterations += 1
            fl_cam = None
            while fl_cam is None:
                sampled_indices = np.random.choice(self.max_num_supporters, RANSAC._MinimalSetSize, replace=False)
                sampled_landmarks = self._landmarks[sampled_indices]
                sampled_features = self._front_features[sampled_indices, :2]  # take only left camera's features
                fl_cam = pnp(points=sampled_landmarks, pixels=sampled_features)
            assert isinstance(fl_cam, Camera)
            curr_supporting_indices = self.__find_supporting_indices(fl_cam)
            if len(curr_supporting_indices) > len(best_supporting_indices):
                best_supporting_indices = curr_supporting_indices
                num_supporters = len(best_supporting_indices)
                self._outlier_probability = 1 - num_supporters / self.max_num_supporters
                self._remaining_iterations = self.__calculate_remaining_iterations()
            else:
                self._remaining_iterations -= 1
        return best_supporting_indices

    def _refine_model(self, best_supporting_indices: np.ndarray) -> Tuple[np.ndarray, Camera]:
        """
        Second loop of RANSAC algorithm:
        1) Use the supporting indices found in the 1st loop, to calculate (PnP) the front-left Camera. Then calculate
            the front-right Camera using the left-to-right transformation.
        2) Project all 3D landmarks onto both Cameras' planes, and calculate the distance between projection and
            actual keypoint. Landmarks with distance lower than {2} on both cameras are considered supporters
            for these Cameras.
        3) Use new set of supporters to re-calculate the front-left Camera, and rerun the iteration.
            Stop iterating when there are no more new supporters.
        :param best_supporting_indices: np array of indices, representing landmarks that were considered supporters in
                the 1st loop of RANSAC.
        :return:
            - best_supporting_indices: np array of indices, representing the supporting landmarks for the best Camera
            - fl_cam: Camera; the front-left Camera calculated using PnP on the set of supporters
        """
        while True:
            self._performed_iterations += 1
            supporting_landmarks = self._landmarks[best_supporting_indices]
            supporting_features = self._front_features[best_supporting_indices]
            fl_cam = pnp(points=supporting_landmarks, pixels=supporting_features[:, :2])  # use only left pixels
            assert isinstance(fl_cam, Camera)
            curr_supporting_indices = self.__find_supporting_indices(fl_cam)
            if len(curr_supporting_indices) > len(best_supporting_indices):
                # we can refine the model even further in the next iteration
                best_supporting_indices = curr_supporting_indices
            else:
                # this is the best model we can find, exit loop
                break
        # finished - return supporters
        return best_supporting_indices, fl_cam

    def __find_supporting_indices(self, fl_cam: Camera) -> np.ndarray:
        """
        Find indices of landmarks that are projected onto the (approximately) correct pixel in both Camera's planes
        :param fl_cam: front-left Camera, as calculated using PnP
        :return: np.ndarray of indices
        """
        left_projections = project(fl_cam, self._landmarks)  # output shape N×2
        left_dists = np.linalg.norm(left_projections.T - self._front_features[:, :2].T, ord=2, axis=0)
        left_supporting_indices = np.where(left_dists <= RANSAC._SupporterThreshold)[0]

        right_projections = project(fl_cam.get_right_camera(), self._landmarks)  # output shape N×2
        right_dists = np.linalg.norm(right_projections.T - self._front_features[:, 2:].T, ord=2, axis=0)
        right_supporting_indices = np.where(right_dists <= RANSAC._SupporterThreshold)[0]

        supporting_indices = np.intersect1d(left_supporting_indices, right_supporting_indices)
        return supporting_indices

    def __calculate_remaining_iterations(self) -> int:
        nominator = np.log(1 - self._success_probability)
        good_set_prob = np.power(1 - self._outlier_probability, self._MinimalSetSize)
        denominator = np.log(1 - good_set_prob)
        res = int(nominator / denominator) + 1
        return min(res, RANSAC._MaxIterationCount)



