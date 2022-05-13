import cv2

from models.frame import Frame
from models.match import MutualMatch
from typing import List


class Matcher:

    Ratio = 0.75

    def __init__(self, matcher_name: str, cross_check=True):
        self.cross_check = cross_check
        self.type = matcher_name.upper()
        self._matcher = self.__create_matcher()

    def match(self, query_desc, train_desc) -> List[cv2.DMatch]:
        matches_2nn = self._matcher.knnMatch(query_desc, train_desc, 2)
        good_matches = [first for (first, second) in matches_2nn if first.distance / second.distance <= self.Ratio]
        return good_matches

    def match_between_frames(self, back_frame: Frame, front_frame: Frame) -> List[MutualMatch]:
        between_frame_matches = self.match(back_frame.inlier_descriptors, front_frame.inlier_descriptors)
        back_frame.next_frame_match_count = len(between_frame_matches)
        mutual_matches = []
        for m in between_frame_matches:
            back_match = back_frame.inlier_matches[m.queryIdx]
            front_match = front_frame.inlier_matches[m.trainIdx]
            mutual_matches.append(MutualMatch(back_match, front_match))
        return mutual_matches

    def __create_matcher(self):
        # create a cv2.matcher object
        if self.type == "BF":
            return cv2.BFMatcher(norm=cv2.NORM_L2, cross_check=self.cross_check)
        if self.type == "FLANN":
            if self.cross_check:
                # TODO
                raise NotImplementedError(f"We currently do not support cross-check with {str(self)}")
            return cv2.FlannBasedMatcher(indexParams=dict(algorithm=0, trees=5), searchParams=dict(checks=50))
        raise NotImplementedError(f"We currently do not support the \"{self.type}\" Matcher")

    def __str__(self):
        return f"{self.type}Matcher"
