import gtsam
from typing import Iterable, List

from final_project.models.Camera import Camera


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

