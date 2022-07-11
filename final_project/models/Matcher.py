import cv2
import numpy as np
from typing import Tuple, List

import final_project.utils as u


class Matcher:
    _2NN_Ratio: float = 0.75
    _InlierVerticalThreshold = 1

    def __init__(self, detector_type: str, matcher_type: str, use_crosscheck=False, use_2nn=False):
        self._use_2nn = use_2nn
        self._detector = self.__create_cv2_detector(detector_type)
        self._matcher = self.__create_cv2_matcher(matcher_type, detector_type, use_crosscheck)

    def match_within_frame(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect cv2.KeyPoint objects on each image and match between them, either by "regular" matching or by
            detecting 2NN matches and only picking the ones that are above a given threshold (KNN_Ratio).
        Then, filter out matches that are not valid for stereo-corrected Frames.

        Returns two numpy arrays that should have the same amount of lines:
            features - array of shape (N, 4) where each line is a match and each column is the X/Y coordinate on either image
            descriptors - array of shape (N,) containing the image1 descriptors of each feature
        """
        img_l, img_r = u.read_images(idx)
        kps_l, descs_l = self._detector.detectAndCompute(img_l, None)
        kps_r, descs_r = self._detector.detectAndCompute(img_r, None)
        matched_indices = self.match_descriptors(descs_l, descs_r)
        features_list, descriptors_list = [], []
        for idx_l, idx_r in matched_indices:
            kp_l, kp_r = kps_l[idx_l], kps_r[idx_r]
            xl, yl = kp_l.pt
            xr, yr = kp_r.pt
            if abs(yl - yr) >= Matcher._InlierVerticalThreshold:
                # this match is not on the Epi-Polar line - ignore it
                continue
            if xl <= xr:
                # this match is triangulated *behind* the cameras - ignore it
                continue
            features_list.append([xl, yl, xr, yr])
            descriptors_list.append(descs_l[idx_l])
        features = np.array(features_list)
        descriptors = np.array(descriptors_list)
        assert features.shape[0] == descriptors.shape[0], "Error extracting matching features"
        return features, descriptors

    def match_descriptors(self, descriptors1, descriptors2) -> List[Tuple[int, int]]:
        """
        Returns a list of pairs, each pair has the indices of the matched descriptors between the two input lists.
        @:raises AssertionError if the number of matched indices do not equal.
        """
        matches = self._match_2nn(descriptors1, descriptors2) if self._use_2nn else self._matcher.match(descriptors1, descriptors2)
        matched_indices1 = [m.queryIdx for m in matches]  # indices of matches from $features1
        matched_indices2 = [m.trainIdx for m in matches]  # indices of matches from $features2
        assert len(matched_indices1) == len(matched_indices2), "match counts are not equal"
        return list(zip(matched_indices1, matched_indices2))

    def _match_2nn(self, query_desc, train_desc):
        matches_2nn = self._matcher.knnMatch(query_desc, train_desc, 2)
        good_matches = [first for (first, second) in matches_2nn if
                        first.distance / second.distance <= Matcher._2NN_Ratio]
        return good_matches

    @staticmethod
    def __create_cv2_detector(detector_type: str):
        detector_type = detector_type.upper()
        if detector_type == "ORB":
            return cv2.ORB_create()
        if detector_type == "SIFT":
            return cv2.SIFT_create()
        raise NotImplementedError(f"{detector_type} Detector is not supported")

    @staticmethod
    def __create_cv2_matcher(matcher_type: str, detector_type: str, use_crosscheck):
        matcher_type = matcher_type.upper()
        detector_type = detector_type.upper()
        if matcher_type == "BF":
            norm = cv2.NORM_HAMMING if detector_type == "ORB" else cv2.NORM_L2
            return cv2.BFMatcher(norm, crossCheck=use_crosscheck)
        if matcher_type == "FLANN":
            if use_crosscheck:
                # TODO
                raise NotImplementedError(f"{matcher_type} Matcher with Cross-Check is currently not supported")
            if detector_type == "ORB":
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            else:
                index_params = dict(algorithm=0, trees=5)
            return cv2.FlannBasedMatcher(indexParams=index_params, searchParams=dict(checks=50))
        raise NotImplementedError(f"{matcher_type} Matcher is currently not supported")
