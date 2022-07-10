import cv2

from final_project.models.Frame import Frame


class Matcher:

    def __init__(self, detector_type: str, matcher_type: str, use_crosscheck: bool = False, knn_ratio=0.75):
        self.knn_ratio = knn_ratio
        self._detector = self.__create_cv2_detector(detector_type)
        self._matcher = self.__create_cv2_matcher(matcher_type, detector_type, use_crosscheck)

    def match_within_frame(self, img1, img2):
        kps1, desc1 = self._detector.detectAndCompute(img1, None)
        kps2, desc2 = self._detector.detectAndCompute(img2, None)
        return None

    def _match_2nn(self, query_desc, train_desc):
        matches_2nn = self._matcher.knnMatch(query_desc, train_desc, 2)
        good_matches = [first for (first, second) in matches_2nn if first.distance / second.distance <= self.knn_ratio]
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
