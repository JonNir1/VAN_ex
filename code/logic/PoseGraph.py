import time
import gtsam
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from models.directions import Side
from models.database import DataBase
from models.camera import Camera
from models.frame import Frame
from models.match import MutualMatch
from models.Graph import Graph
from models.matcher import Matcher
from models.gtsam_frame import GTSAMFrame
from logic.Bundle2 import Bundle2
from logic.ransac import Ransac
from logic.db_adapter import DBAdapter


class PoseGraph:
    KeyframeDistanceThreshold = 10
    MahalanobisThreshold = 2.0
    MatchCountThreshold = 100
    OutlierPercentThreshold = 20.0

    # for some reason, loop matching works poorly with BF matcher (the default we use), but works
    # fast and well with these parameters
    _Loop_Matcher = Matcher(matcher_name="flann", cross_check=False, use_2nn=True)

    def __init__(self, bundles: List[Bundle2]):
        self.keyframe_symbols: Dict[int, int] = dict()
        self._is_optimized = False
        self._initial_estimates = gtsam.Values()
        self._optimized_estimates = gtsam.Values()
        self._factor_graph = gtsam.NonlinearFactorGraph()
        self._locations_graph = Graph()
        self._preprocess_bundles(bundles)

    @property
    def error(self):
        if self._is_optimized:
            return self._factor_graph.error(self._optimized_estimates)
        return self._factor_graph.error(self._initial_estimates)

    def optimize_with_loops(self, max_loops_to_close: Optional[int] = None, verbose=False):
        start_time, minutes_counter = time.time(), 0
        if verbose:
            print("\nStarting loop closure...")

        optimizer = gtsam.LevenbergMarquardtOptimizer(self._factor_graph, self._initial_estimates)
        intermediate_results = optimizer.optimize()

        keyframe_indices = sorted(self.keyframe_symbols.keys())
        max_loops_to_close = len(keyframe_indices) + 1 if max_loops_to_close is None else max_loops_to_close

        closed_loops_count = 0
        for i, front_idx in enumerate(keyframe_indices):
            front_symbol = self.keyframe_symbols[front_idx]
            for j in range(i - self.KeyframeDistanceThreshold):  # do not match bundles that are too close to each other
                if closed_loops_count > max_loops_to_close:
                    break
                curr_minute = int((time.time() - start_time) / 60)
                if verbose and curr_minute > minutes_counter:
                    minutes_counter = curr_minute
                    print(f"\tElapsed Minutes:\t{minutes_counter}\n\tCurrent Keyframe:\t{i}\n")

                back_idx = keyframe_indices[j]
                if not self._is_within_mahalanobis_range(intermediate_results, front_idx, back_idx):
                    # this pair is not a possible loop, continue to next pair
                    continue
                back_frame, front_frame, matches, supporters = self._match_possible_loop(front_idx, back_idx)
                outlier_percent = 100 * (len(matches) - len(supporters)) / len(matches)
                if outlier_percent > self.OutlierPercentThreshold:
                    # there are not enough supporters to justify loop on this pair
                    continue

                # need to add this as constraint to Factor Graph & as edge Locations Graph
                if verbose:
                    print(f"Loop #{closed_loops_count + 1}")
                    print(f"\tFrame{front_idx}\t<-->\tFrame{back_idx}")
                relative_pose, relative_cov = self._calculate_loop_relative_pose(back_frame, front_frame,
                                                                                 supporters)
                noise_model = gtsam.noiseModel.Gaussian.Covariance(relative_cov)
                back_symbol = self.keyframe_symbols[back_idx]
                factor = gtsam.BetweenFactorPose3(back_symbol, front_symbol, relative_pose, noise_model)
                self._factor_graph.add(factor)

                back_vertex_id = self._locations_graph.get_vertex_id(frame_idx=back_idx, symbol=back_symbol)
                front_vertex_id = self._locations_graph.get_vertex_id(frame_idx=front_idx, symbol=front_symbol)
                self._locations_graph.create_or_update_edge(v1_id=back_vertex_id, v2_id=front_vertex_id, cov=relative_cov)

                # TODO: instead of optimizing on each loop, optimize every 5 KFs / 5 loops
                prev_err = self._factor_graph.error(intermediate_results)
                optimizer = gtsam.LevenbergMarquardtOptimizer(self._factor_graph, intermediate_results)
                intermediate_results = optimizer.optimize()
                curr_err = self._factor_graph.error(intermediate_results)
                closed_loops_count = closed_loops_count + 1

                if verbose:
                    print(f"\tOutlier Percent:\t{outlier_percent:.2f}%")
                    print(f"\tError Difference:\t{prev_err - curr_err}\n")

        # final optimization just to be sure
        optimizer = gtsam.LevenbergMarquardtOptimizer(self._factor_graph, intermediate_results)
        self._optimized_estimates = optimizer.optimize()
        self._is_optimized = True

    def optimize_without_loops(self):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self._factor_graph, self._initial_estimates)
        self._optimized_estimates = optimizer.optimize()
        self._is_optimized = True

    def extract_relative_cameras(self) -> pd.Series:
        if not self._is_optimized:
            raise RuntimeError("Cannot extract cameras from non-optimized PoseGraph")

        # first keyframe
        start_symbol = self.keyframe_symbols[0]
        start_pose = self._optimized_estimates.atPose3(start_symbol)
        start_cam = Camera.from_pose3(idx=0, pose=start_pose)

        relative_cameras = [start_cam]
        prev_cam = start_cam
        for (fr_idx, fr_sym) in self.keyframe_symbols.items():
            if fr_idx == 0:
                # already handled the first keyframe
                continue
            pose = self._optimized_estimates.atPose3(fr_sym)
            abs_cam = Camera.from_pose3(fr_idx, pose)
            prev_R, prev_t = prev_cam.get_rotation_matrix(), prev_cam.get_translation_vector()
            curr_R, curr_t = abs_cam.get_rotation_matrix(), abs_cam.get_translation_vector()
            R = curr_R @ prev_R.T
            t = curr_t - R @ prev_t
            relative_cameras.append(Camera(idx=fr_idx, side=Side.LEFT, extrinsic_mat=np.hstack([R, t])))
        s = pd.Series(data=relative_cameras, index=[cam.idx for cam in relative_cameras], name="pg_cameras")
        s.index.name = DataBase.FRAMEIDX
        return s

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
        self._locations_graph.create_vertex(frame_idx=0, symbol=kf0_symbol)

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

            # add current keyframe to the locations-graph, with weighted edge to previous keyframe:
            v = self._locations_graph.create_vertex(frame_idx=keyframe_idx, symbol=keyframe_symbol)
            e = self._locations_graph.create_or_update_edge(v1_id=v.index - 1, v2_id=v.index, cov=relative_cov)

            # update values for next iteration:
            prev_symbol = keyframe_symbol
            prev_cam = global_camera
        return None

    def _is_within_mahalanobis_range(self, current_values: gtsam.Values, front_idx: int, back_idx: int) -> bool:
        # calculate Delta C:
        front_symbol = self.keyframe_symbols[front_idx]
        front_pose = current_values.atPose3(front_symbol)
        back_symbol = self.keyframe_symbols[back_idx]
        back_pose = current_values.atPose3(back_symbol)
        relative_pose = back_pose.between(front_pose)
        delta_cam = np.hstack([relative_pose.rotation().xyz(), relative_pose.translation()])

        # calculate Cov matrix:
        front_vertex = self._locations_graph.get_vertex_id(front_idx, front_symbol)
        back_vertex = self._locations_graph.get_vertex_id(back_idx, back_symbol)
        cov_matrix = self._locations_graph.get_relative_covariance(back_vertex, front_vertex)

        mahalanobis_distance = delta_cam @ cov_matrix @ delta_cam
        return 0 < mahalanobis_distance <= self.MahalanobisThreshold

    @staticmethod
    def _match_possible_loop(front_idx: int, back_idx: int):
        default_camera_left = Camera.create_default()
        default_camera_left.idx = back_idx  # for indexing consistency
        default_camera_right = default_camera_left.calculate_right_camera()
        back_frame = Frame(idx=back_idx, left_cam=default_camera_left, right_cam=default_camera_right)
        front_frame = Frame(front_idx)
        matches = PoseGraph._Loop_Matcher.match_between_frames(back_frame, front_frame)
        if len(matches) < PoseGraph.MatchCountThreshold:
            # not enough matches between candidates - exit early
            return back_frame, front_frame, matches, []
        fl_cam, fr_cam, supporters = Ransac().run(matches, bl_cam=back_frame.left_camera, br_cam=back_frame.right_camera)
        fl_cam.idx = front_idx  # for indexing consistency
        fr_cam.idx = front_idx  # for indexing consistency
        front_frame.left_camera = fl_cam
        front_frame.right_camera = fr_cam
        return back_frame, front_frame, matches, supporters

    @staticmethod
    def _calculate_loop_relative_pose(back_frame: Frame, front_frame: Frame, supporters: List[MutualMatch]):
        # build the bundle
        track_id = 0
        for sup in supporters:
            back_match, front_match = sup.back_frame_match, sup.front_frame_match
            back_frame.match_to_track_id[back_match] = track_id
            front_frame.match_to_track_id[front_match] = track_id
            track_id = track_id + 1
        dba = DBAdapter([back_frame, front_frame])
        b = Bundle2(dba.cameras_db[DataBase.CAM_LEFT], dba.tracks_db)

        # compute pose and covariance
        b.adjust()
        relative_pose = b.calculate_keyframes_relative_pose()  # relative to $back_frame
        relative_cov = b.calculate_keyframes_relative_covariance()
        return relative_pose, relative_cov

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
