import os
import cv2
import math
import gtsam
from typing import Iterable, List

import final_project.config as c
from final_project.models.Camera import Camera


def read_images(idx: int):
    """
    Load a pair of KITTI images with the given index
    """
    image_name = "{:06d}.png".format(idx)
    left_dir = "image_0"
    left_path = os.path.join(c.DATA_READ_PATH, "sequences", "00", left_dir, image_name)
    left_image = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)

    right_dir = "image_1"
    right_path = os.path.join(c.DATA_READ_PATH, "sequences", "00", right_dir, image_name)
    right_image = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    return left_image, right_image


def convert_to_absolute_cameras(cams: Iterable[Camera]) -> List[Camera]:
    abs_cams: List[Camera] = []
    for i, rel_cam in enumerate(cams):
        if i == 0:
            abs_cams.append(rel_cam)
        else:
            prev_cam = abs_cams[-1]
            prev_R, prev_t = prev_cam.R, prev_cam.t
            curr_R_rel, curr_t_rel = rel_cam.R, rel_cam.t
            R = curr_R_rel @ prev_R
            t = curr_t_rel + curr_R_rel @ prev_t
            abs_cams.append(Camera.from_Rt(R, t))
    return abs_cams


def convert_to_relative_cameras(cams: Iterable[Camera]) -> List[Camera]:
    relative_cameras = []
    for i, abs_cam in enumerate(cams):
        if i == 0:
            relative_cameras.append(abs_cam)
            prev_cam = abs_cam
        else:
            prev_R, prev_t = prev_cam.R, prev_cam.t
            curr_R, curr_t = abs_cam.R, abs_cam.t
            R_rel = curr_R @ prev_R.T
            t_rel = curr_t - R_rel @ prev_t
            rel_cam = Camera.from_Rt(R_rel, t_rel)
            relative_cameras.append(rel_cam)
            prev_cam = abs_cam
    return relative_cameras


def calculate_gtsam_pose(cam: Camera) -> gtsam.Pose3:
    R, t = cam.R, cam.t
    gtsam_R = gtsam.Rot3(R.T)
    gtsam_t = gtsam.Point3((- R.T @ t).reshape((3,)))
    return gtsam.Pose3(gtsam_R, gtsam_t)


def calculate_gtsam_stereo_params() -> gtsam.Cal3_S2Stereo:
    Camera.init_class_attributes()  # TODO: fixme - no external init
    K = Camera.K()
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    skew = K[0][1]
    baseline = - Camera._RightTranslation[0][0]  # TODO: fixme
    gtsam_stereo_params = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, baseline)
    return gtsam_stereo_params


def calculate_camera_from_gtsam_pose(pose: gtsam.Pose3) -> Camera:
    R = pose.rotation().matrix().T
    t = -R @ pose.translation().reshape((3, 1))
    return Camera.from_Rt(R, t)


def choose_keyframe_indices(max_frame_idx: int, bundle_size: int = c.BUNDLE_SIZE):
    """
    Returns a list of integers representing the keypoint indices, where each Bundle is of size $bundle_size
    see: https://stackoverflow.com/questions/72292581/split-list-into-chunks-with-repeats-between-chunks
    """
    # TODO: enable other methods to choose keyframes (e.g. #Tracks, distance travelled, etc.)
    interval = bundle_size - 1
    all_idxs = list(range(max_frame_idx + 1))
    bundled_idxs = [all_idxs[i * interval: i * interval + bundle_size] for i in
                    range(math.ceil((len(all_idxs) - 1) / interval))]
    # return bundled_idxs  # if we want to return a list-of-list containing all indices in each bundle
    keypoint_idxs = [bundle[0] for bundle in bundled_idxs]
    if keypoint_idxs[-1] == all_idxs[-1]:
        return keypoint_idxs
    keypoint_idxs.append(all_idxs[-1])
    return keypoint_idxs
