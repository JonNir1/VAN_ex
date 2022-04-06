"""
Representing a quadrant (4) of KITTI images:
  - Two stereo-rectified pairs of images with adjacent index
     (e.g. [img0_l, img0_r] & [img1_l, img1_r]
  - Single tracking image pair from the left images
     (e.g. [img0_l, img1_l]

The Quadrant's inliers are based on the Consensus match, meaning matches that are agreed upon by all 3 couples
"""

from StereoImagePair import StereoImagePair
from TrackingImagePair import TrackingImagePair
from .. import utils as u


class TrackingQuadrant:

    def __init__(self, lower_idx):
        self.lower_idx = lower_idx
        self.back_pair = StereoImagePair(lower_idx)
        self.front_pair = StereoImagePair(lower_idx + 1)
        self.tracking_pair = TrackingImagePair(lower_idx)
        self.consensus_matches = self.__extract_consensus_matching()

    def __extract_consensus_matching(self):
        consensus_matches = []
        back_inliers = self.back_pair.inlier_matches
        # check which of the $back_inliers is in the tracking inliers
        # TODO
        raise NotImplementedError("TrackingQuadrant class is still not implemented")



