import utils as u
import config as c

from models.track import Track


class Frame:
    """
    Represents a stereo rectified image-pair from the KITTI dataset
    """

    MaxIndex = 3449
    MaxVerticalDistance = 1

    def __init__(self, idx: int):
        if idx < 0:
            raise IndexError(f"Frame index must be between 0 and {self.MaxIndex}, not {idx}")
        self.idx = idx
        self._tracks = dict()  # a dict matching Track to (kp_left, kp_right)

    def get_idx(self):
        return self.idx

    def get_tracks(self):
        return self._tracks

    def detect_and_match(self, **kwargs):
        """
        Detects keypoints and extracts matches between the two images.
        A match is only extracted if the two keypoints have a horizontal distance below a given number of pixels.

        optional params:
            $detector_name: str -> name of the detector to use, default "sift"
            $matcher_name: str -> name of the matcher to use, default "flann"
            $max_vertical_distance: positive int -> threshold for identifying bad matches

        returns the keypoints & their descriptors, as well as the inlier matches
        """
        img_left, img_right = u.read_image_pair(self.idx)

        # detect keypoints in both images of the frame
        detector_name = kwargs.get("detector_name", c.DEFAULT_DETECTOR_NAME)
        detector = u.create_detector(detector_name)
        keypoints_left, descriptors_left = detector.detectAndCompute(img_left, None)
        keypoints_right, descriptors_right = detector.detectAndCompute(img_right, None)

        # find matches between images, and exclude outliers
        matcher_name = kwargs.get("matcher_name", c.DEFAULT_MATCHER_NAME)
        matcher = u.create_matcher(matcher_name)
        all_matches = matcher.match(descriptors_left, descriptors_right)

        # exclude outlier matches
        max_vertical_distance = kwargs.get("max_vertical_distance", self.MaxVerticalDistance)
        inlier_matches = []
        for m in all_matches:
            kp_left = keypoints_left[m.queryIdx]
            kp_right = keypoints_right[m.trainIdx]
            vertical_dist = abs(kp_left.pt[1] - kp_right.pt[1])
            if vertical_dist <= max_vertical_distance:
                inlier_matches.append(m)
        return keypoints_left, descriptors_left, keypoints_right, descriptors_right, inlier_matches

    def __str__(self):
        return f"Frame{self.idx}"

