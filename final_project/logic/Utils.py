import os
import math
import enum
import numpy as np
from typing import Iterable, List

import final_project.config as c
from final_project.models.Camera import Camera


class Axis(enum.Enum):
    X = 0
    Y = 1
    Z = 2
    All = 3


class EulerAngle(enum.Enum):
    YAW = 0
    PITCH = 1
    ROLL = 2


def read_ground_truth_cameras(use_relative=False) -> List[Camera]:
    """
    Reads KITTI's ground-truth Cameras in absolute coordinates (i.e., first Camera's location)
    If $use_relative is True, convert the cameras to relative coordinates (relative to previous Camera's location)
    """
    path = os.path.join(c.DATA_READ_PATH, 'poses', '00.txt')
    cameras = []
    f = open(path, 'r')
    for i, line in enumerate(f.readlines()):
        mat = np.array(line.split(), dtype=float).reshape((3, 4))
        R, t = mat[:, :3], mat[:, 3:]
        cam = Camera.from_Rt(R, t)
        cameras.append(cam)
    if use_relative:
        return convert_to_relative_cameras(cameras)
    return cameras


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
