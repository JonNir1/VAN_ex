import gtsam

from final_project.models.Camera import Camera


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

