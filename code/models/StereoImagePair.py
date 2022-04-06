"""
Represents a pair of KITTY images with the same index,
i.e. stereo rectified images
"""

from . import ImagePairBase
from .. import utils as u


class StereoImagePair(ImagePairBase):

    MaxVerticalDistance = 1  # units are pixels

    def __init__(self, idx: int):
        img_l, img_r = u.read_image_pair(idx)
        super().__init__(img_l, img_r)
        self.idx = idx

    def _identify_inliers(self, matches) -> (list, list):
        inliers, outliers = [], []
        for match in matches:
            vertical_dist = self.__calculate_vertical_distance(match)
            if vertical_dist <= self.MaxVerticalDistance:
                inliers.append(match)
            else:
                outliers.append(match)
        return inliers, outliers

    def __calculate_vertical_distance(self, single_match) -> float:
        kp_left = self.keypoints0[single_match.queryIdx]
        kp_right = self.keypoints1[single_match.trainIdx]
        return abs(kp_left.pt[1] - kp_right.pt[1])
