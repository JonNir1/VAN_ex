from typing import Optional

import cv2

import config as c
import utils as u
from models.frame import Frame
from models.match import FrameMatch, MutualMatch

MaxVerticalDistance = 1


def detect_and_match(frame: Frame, **kwargs):
    """
    Detects keypoints and extracts matches between two images of a single Frame.
    A match is only extracted if the two keypoints have a horizontal distance below a given number of pixels.

    optional params:
    $detector_name: str -> name of the detector to use, default "sift"
    $matcher_name: str -> name of the matcher to use, default "flann"
    $max_vertical_distance: positive int -> threshold for identifying bad matches

    returns the keypoints & their descriptors, as well as the inlier matches
    """
    img_left, img_right = u.read_image_pair(frame.get_id())

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
    max_vertical_distance = kwargs.get("max_vertical_distance", MaxVerticalDistance)
    inlier_matches = []
    for m in all_matches:
        kp_left = keypoints_left[m.queryIdx]
        kp_right = keypoints_right[m.trainIdx]
        vertical_dist = abs(kp_left.pt[1] - kp_right.pt[1])
        if vertical_dist <= max_vertical_distance:
            inlier_matches.append(m)
    return keypoints_left, descriptors_left, keypoints_right, descriptors_right, inlier_matches


def match_between_frames(back_frame: Frame, front_frame: Frame, **kwargs) -> (list[MutualMatch], list[FrameMatch]):
    """
    Detect and match keypoints within two subsequent Frames, and then match between the Frames.

    optional params:
    $detector_name: str -> name of the detector to use, default "sift"
    $matcher_name: str -> name of the matcher to use, default "flann"
    $max_vertical_distance: positive int -> threshold for identifying bad matches

    Returns two lists:
    - mutual_matches: A list of 4-tuples, matching four keypoints from each of the 4 images
    - unique_matches: A list of 2-tuples, containing keypoints of the front-matches that don't have a corresponding
        match in among the back-matches.
    """
    detector_name = kwargs.get("detector_name", c.DEFAULT_DETECTOR_NAME)
    matcher_name = kwargs.get("matcher_name", c.DEFAULT_MATCHER_NAME)
    max_vertical_distance = kwargs.get("max_vertical_distance", MaxVerticalDistance)

    # detect keypoints and extract inliers for each Frame separately
    kps_back_left, desc_back_left, kps_back_right, _, back_matches = detect_and_match(back_frame,
                                                                                      detector_name=detector_name,
                                                                                      matcher_name=matcher_name,
                                                                                      max_vertical_distance=max_vertical_distance)
    kps_front_left, desc_front_left, kps_front_right, _, front_matches = detect_and_match(front_frame,
                                                                                          detector_name=detector_name,
                                                                                          matcher_name=matcher_name,
                                                                                          max_vertical_distance=max_vertical_distance)
    # extract matches between frames
    matcher = u.create_matcher(matcher_name)
    between_frame_matches = matcher.match(desc_back_left, desc_front_left)

    # find keypoints that are matched within both Frame and between Frames:
    mutual_matches_cv2 = _find_mutual_matches(back_matches, front_matches, between_frame_matches)
    back_mutual_matches = [FrameMatch(kps_back_left[back_match_cv2.queryIdx],
                                      kps_back_right[back_match_cv2.trainIdx])
                           for (back_match_cv2, _) in mutual_matches_cv2]
    front_mutual_matches = [FrameMatch(kps_front_left[front_match_cv2.queryIdx],
                                       kps_front_right[front_match_cv2.trainIdx])
                            for (_, front_match_cv2) in mutual_matches_cv2]
    mutual_matches = [MutualMatch(back_match, front_match)
                      for (back_match, front_match) in zip(back_mutual_matches, front_mutual_matches)]

    # extract front-matches that don't have a corresponding back-match -> these are new Tracks
    front_matches_with_back_match = [fm for (bm, fm) in mutual_matches_cv2]
    front_keypoints_without_back_match = [FrameMatch(kps_front_left[m.queryIdx], kps_front_right[m.trainIdx])
                                          for m in front_matches if m not in front_matches_with_back_match]
    return mutual_matches, front_keypoints_without_back_match


def _find_mutual_matches(back_matches: list, front_matches: list,
                         between_frames_matches: list) -> list[tuple[cv2.DMatch, cv2.DMatch]]:
    # find $front_matches that have a $back_match that corresponds to them (via a $between_match)
    mutual_matches = []
    back_matches_left_idxs = [m.queryIdx for m in back_matches]
    front_matches_left_idxs = [m.queryIdx for m in front_matches]
    for match_between in between_frames_matches:
        try:
            back_match_idx = back_matches_left_idxs.index(match_between.queryIdx)
            front_match_idx = front_matches_left_idxs.index(match_between.trainIdx)
            mutual_matches.append((back_matches[back_match_idx], front_matches[front_match_idx]))
        except ValueError:
            continue
    return mutual_matches


def _find_within_frame_matches(between_frames_match, back_matches_left_indices: list,
                               front_matches_left_indices: list) -> tuple[Optional[int], Optional[int]]:
    """
    For a match between two keypoints on the Frames' left images, we find matches within each frame that contain these
    two keypoints, and return these matches' indices or None if no such match exists.
    """
    # attempt to find a back_frame_match with the same left kp as $between_frames_match's back kp
    try:
        index_of_back_kp_in_back_match = back_matches_left_indices.index(between_frames_match.queryIdx)
    except ValueError:
        index_of_back_kp_in_back_match = None

    # attempt to find a front_frame_match with the same left kp as $between_frames_match's front kp
    try:
        index_of_front_kp_in_front_matches = front_matches_left_indices.index(between_frames_match.trainIdx)
    except ValueError:
        # front_matches doesn't contain this keypoint - this is not a consensus match
        index_of_front_kp_in_front_matches = None
    return index_of_back_kp_in_back_match, index_of_front_kp_in_front_matches

