import gtsam
from typing import NamedTuple
from itertools import count

from models.directions import Side
from models.camera import Camera


class GTSAMFrame(NamedTuple):

    symbol: int
    pose: gtsam.Pose3
    stereo_params: gtsam.Cal3_S2Stereo
    _counter = count(0)

    @staticmethod
    def from_camera(cam: Camera):
        assert cam.side == Side.LEFT, "must use left camera"
        symbol = gtsam.symbol('c', next(GTSAMFrame._counter))
        pose = GTSAMFrame._calculate_pose(cam)
        stereo_params = GTSAMFrame._extract_stereo_params(cam)
        return GTSAMFrame(symbol, pose, stereo_params)

    @staticmethod
    def _calculate_pose(cam: Camera) -> gtsam.Pose3:
        """
        Converts a Camera object to a gtsam.Pose3 object.
        Camera objects map a 3D point in GLOBAL coordinates (x), and project them to CAMERA coord: C_x=[R|t]@[x|1].T
        Pose3 objects map from CAMERA coordinates to GLOBAL coord. Thus we need to re-calculate the rotation-matrix and
        translation-vector to match gtsam assumptions:
            gtsam_Rot = R^-1 = R.T ; gtsam_trans = - R.T @ t
        """
        R, t = cam.get_rotation_matrix(), cam.get_translation_vector()
        gtsam_R = gtsam.Rot3(R.T)
        gtsam_t = gtsam.Point3((- R.T @ t).reshape((3,)))
        return gtsam.Pose3(gtsam_R, gtsam_t)

    @staticmethod
    def _extract_stereo_params(cam: Camera):
        """
        Using the Camera class attributes (_K, _RightTranslation), creates a gtsam.Cal3_S2Stereo object.
        These objects hold all required parameters to represent a stereo-camera model (except the actual camera's [R|t])
        """
        K = cam.intrinsic_matrix
        fx, fy = K[0][0], K[1][1]
        cx, cy = K[0][2], K[1][2]
        skew = K[0][1]
        baseline = - cam.right_translation[0][0]
        gtsam_stereo_params = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, baseline)
        return gtsam_stereo_params

