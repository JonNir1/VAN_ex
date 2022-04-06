"""
Representing a pair of KITTY images from the left camera and adjacent indices,
e.g. img0_left, img1_left (the lower index should always come first)
"""

from . import ImagePairBase
from .. import utils as u


class TrackingImagePair(ImagePairBase):

    def __init__(self, lower_idx: int):
        img0_l = u.read_single_image(lower_idx, False)
        img1_l = u.read_single_image(lower_idx + 1, False)
        super().__init__(img0_l, img1_l)
        self.lower_idx = lower_idx

    def _identify_inliers(self, matches) -> (list, list):
        return matches, []
