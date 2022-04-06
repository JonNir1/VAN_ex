import cv2
from abc import ABC, abstractmethod
from .. import utils as u


class ImagePairBase(ABC):

    def __init__(self, img0, img1):
        self.img0, self.img1 = img0, img1
        self.keypoints0, self.keypoints1 = None, None
        self.inlier_matches, self.outlier_matches = None, None

    @property
    def images(self):
        return self.img0, self.img1

    @images.setter
    def images(self, img0, img1):
        if img0.size == 0 or img1.size == 0:
            raise RuntimeError("could not load the provided images")
        self.img0 = img0
        self.img1 = img1

    def detect_and_match_keypoints(self, detector_name: str, matcher_name: str,
                                   norm=cv2.NORM_L2, cross_check: bool = True):
        detector = u.create_detector(detector_name)
        self.keypoints0, descriptors0 = detector.detectAndCompute(self.img0, None)
        self.keypoints1, descriptors1 = detector.detectAndCompute(self.img1, None)
        matcher = u.create_matcher(matcher_name, norm, cross_check)
        matches = matcher.match(descriptors0, descriptors1)
        self.inlier_matches, self.outlier_matches = self._identify_inliers(matches)

    @abstractmethod
    def _identify_inliers(self, matches) -> (list, list):
        pass
