import time
import random
import numpy as np

from models.directions import Side, Position
from models.match import MutualMatch
from models.camera import Camera
from logic.triangulation import triangulate
from logic.pnp import compute_front_cameras


class Ransac:

    MinimalSetSize = 4
    DefaultSuccessProbability = 0.99
    MaxDistanceForSupporter = 2

    def __init__(self, success_prob: float = DefaultSuccessProbability):
        self._success_probability = success_prob
        self._outlier_probability = 0.99                               # this value is updated during the Ransac's run
        self._num_iterations = self.__calculate_number_of_iteration()  # this value is updated during the Ransac's run


    def run(self, mutual_matches: list[MutualMatch], bl_cam: Camera, br_cam: Camera,
            verbose: bool = False) -> tuple[Camera, Camera, list[MutualMatch]]:
        start = time.time()
        if verbose:
            print(f"Starting RANSAC with {self._num_iterations} iterations")
        matched_points_3d = triangulate([m.get_frame_match(Position.BACK) for m in mutual_matches], bl_cam, br_cam)
        actual_projections_fl = np.array([m.get_keypoint(Side.LEFT, Position.FRONT).pt for m in mutual_matches])
        actual_projections_fr = np.array([m.get_keypoint(Side.RIGHT, Position.FRONT).pt for m in mutual_matches])
        supporting_indices = self._find_supporters_subset(mutual_matches, bl_cam, br_cam, matched_points_3d,
                                                          actual_projections_fl, actual_projections_fr, verbose)
        fl_cam, fr_cam, supporting_matches = self._refine_best_model(mutual_matches, bl_cam, br_cam, supporting_indices,
                                                                     matched_points_3d, actual_projections_fl,
                                                                     actual_projections_fr, verbose)
        elapsed = time.time() - start
        if verbose:
            print(f"Completed RANSAC in {elapsed:.2f} seconds\n\tNumber of Supporters: {len(supporting_matches)}\n")
        return fl_cam, fr_cam, supporting_matches

    def _find_supporters_subset(self, mutual_matches: list[MutualMatch], bl_cam: Camera, br_cam: Camera,
                                matched_points_3d: np.ndarray, front_left_pixels: np.ndarray,
                                front_right_pixels: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        First loop of RANSAC: find a model with as many supporters as possible,
            using a minimal-set (default size 4) of MutualMatch objects.
        In each iteration sample 4 MutualMatch objects and calculate front-left and front-right cameras based on them.
            Then, project all 3D-points to the cameras' planes and see how many are projected within a given distance
            from their real cv2.KeyPoint pixel (default distance 2). Those points are considered supporters.
        Returns the indices of supporting MutualMatches for the best model that was found.
        """
        if verbose:
            print("Starting supporters' loop of RANSAC")
        max_num_supporters = len(mutual_matches)
        best_supporting_indices = np.array([])
        while self._num_iterations > 0:
            cam_fl, cam_fr = None, None
            while cam_fl is None:
                sampled_matches = random.sample(mutual_matches, self.MinimalSetSize)
                cam_fl, cam_fr = compute_front_cameras(sampled_matches, bl_cam, br_cam)

            assert isinstance(cam_fl, Camera) and isinstance(cam_fr, Camera)
            supporting_indices_left = self._estimate_camera(cam_fl, matched_points_3d, front_left_pixels)
            supporting_indices_right = self._estimate_camera(cam_fr, matched_points_3d, front_right_pixels)
            supporting_indices = np.intersect1d(supporting_indices_left, supporting_indices_right)

            if len(supporting_indices) > len(best_supporting_indices):
                best_supporting_indices = supporting_indices
                num_supporters = len(best_supporting_indices)
                self.__update_parameters(num_supporters, max_num_supporters)
                if verbose:
                    print(f"\tUpdated Parameters:\tNum. Supporters: {num_supporters}" +
                          f"\tRem. Iterations: {self._num_iterations}")
            else:
                self._num_iterations -= 1
                if verbose and self._num_iterations % 100 == 0:
                    print(f"\tCurrent Parameters:\tNum. Supporters: {len(best_supporting_indices)}" +
                          f"\tRem. Iterations: {self._num_iterations}")
        return best_supporting_indices

    def _refine_best_model(self, mutual_matches: list[MutualMatch], bl_cam: Camera, br_cam: Camera,
                           best_supporting_indices: np.ndarray, matched_points_3d: np.ndarray,
                           front_left_pixels: np.ndarray, front_right_pixels: np.ndarray,
                           verbose: bool = False) -> tuple[Camera, Camera, list[MutualMatch]]:
        if verbose:
            print("Starting refinement loop of RANSAC")
        while True:
            curr_supporters = [mutual_matches[idx] for idx in best_supporting_indices]
            cam_fl, cam_fr = None, None
            while cam_fl is None:
                cam_fl, cam_fr = compute_front_cameras(curr_supporters, bl_cam, br_cam)
            assert isinstance(cam_fl, Camera) and isinstance(cam_fr, Camera)
            supporting_indices_left = self._estimate_camera(cam_fl, matched_points_3d, front_left_pixels)
            supporting_indices_right = self._estimate_camera(cam_fr, matched_points_3d, front_right_pixels)
            curr_supporting_indices = np.intersect1d(supporting_indices_left, supporting_indices_right)

            if len(curr_supporting_indices) > len(best_supporting_indices):
                # we can refine the model even further in the next iteration
                best_supporting_indices = curr_supporting_indices
            else:
                # this is the best model we can find, exit loop
                break
        # finished - return model & supporters
        final_supporters = [mutual_matches[idx] for idx in best_supporting_indices]
        return cam_fl, cam_fr, final_supporters

    # @classmethod
    # def _build_model(cls, mutual_matches: list[MutualMatch],
    #                  bl_cam: Camera, br_cam: Camera) -> tuple[Camera, Camera]:
    #     fl_cam, fr_cam = None, None
    #     while fl_cam is None:
    #         sampled_matches = random.sample(mutual_matches, cls.MinimalSetSize)
    #         fl_cam, fr_cam = compute_front_cameras(sampled_matches, bl_cam, br_cam)
    #     return fl_cam, fr_cam

    @staticmethod
    def _estimate_camera(cam: Camera, real_points_3d: np.ndarray, actual_projections: np.ndarray,
                         max_distance: int = MaxDistanceForSupporter) -> np.ndarray:
        cloud_shape = real_points_3d.shape
        assert cloud_shape[0] == 3 or cloud_shape[1] == 3, "Argument $real_points_3d is not a 3D-points array"
        if cloud_shape[0] != 3:
            real_points_3d = real_points_3d.T

        projections_shape = actual_projections.shape
        assert projections_shape[0] == 2 or projections_shape[1] == 2,\
            "Argument $actual_projections is not a 2D-points array"
        if projections_shape[0] != 2:
            actual_projections = actual_projections.T

        calculated_projections = cam.project_3d_points(real_points_3d)  # shape 2xN
        assert calculated_projections.shape == actual_projections.shape
        euclidean_distances = np.linalg.norm(actual_projections - calculated_projections, ord=2, axis=0)
        supporting_indices = np.where(euclidean_distances <= max_distance)[0]
        return supporting_indices

    def __calculate_number_of_iteration(self) -> int:
        """
        Calculate how many iterations of RANSAC are required to get good enough results,
        i.e. for a set of size $s, with outlier probability $e and success probability $p
        we need N > log(1-$p) / log(1-(1-$e)^$s)

        :return: N: int -> number of iterations
        """
        nom = np.log(1 - self._success_probability)
        good_set_prob = np.power(1 - self._outlier_probability, self.MinimalSetSize)
        denom = np.log(1 - good_set_prob)
        return int(nom / denom) + 1

    def __update_parameters(self, num_supporters: int, max_supporters: int):
        self._outlier_probability = 1 - num_supporters / max_supporters
        self._num_iterations = self.__calculate_number_of_iteration()
