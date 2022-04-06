import cv2
from abc import ABC, abstractmethod
from .. import config as c


class ImagePairBase(ABC):

    def __init__(self, img0, img1):
        self.img0, self.img1 = img0, img1
        self.keypoints0, self.keypoints1 = None, None
        self.inlier_matches, self.outlier_matches = None, None
        self.__detect_and_match_keypoints()

    @property
    def images(self):
        return self.img0, self.img1

    @images.setter
    def images(self, img0, img1):
        if img0.size == 0 or img1.size == 0:
            raise RuntimeError("could not load the provided images")
        self.img0 = img0
        self.img1 = img1

    @abstractmethod
    def _identify_inliers(self, matches) -> (list, list):
        pass

    def __detect_and_match_keypoints(self):
        detector = c.DEFAULT_DETECTOR
        self.keypoints0, descriptors0 = detector.detectAndCompute(self.img0, None)
        self.keypoints1, descriptors1 = detector.detectAndCompute(self.img1, None)
        matcher = c.DEFAULT_MATCHER
        matches = matcher.match(descriptors0, descriptors1)
        self.inlier_matches, self.outlier_matches = self._identify_inliers(matches)
