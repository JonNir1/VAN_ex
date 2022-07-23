import gtsam
from typing import Optional
from itertools import count

from final_project.models.Camera import Camera


class GtsamFrame:
    _counter = count(0)
    _is_init: bool = False
    _StereoParams: Optional[gtsam.Cal3_S2Stereo] = None

    def __init__(self, idx: int, cam: Camera):
        self.idx = idx
        self._symbol = gtsam.symbol('c', next(GtsamFrame._counter))
        self._pose = self._calculate_pose(cam)

    @staticmethod
    def _calculate_pose(cam: Camera) -> gtsam.Pose3:
        # Converts a Camera to a gtsam.Pose3 object
        # Camera objects map 3D points in GLOBAL coordinates (x) to Camera's coordinates; whereas Pose3 do the opposite
        R, t = cam.R, cam.t
        gtsam_R = gtsam.Rot3(R.T)
        gtsam_t = gtsam.Point3((- R.T @ t).reshape((3,)))
        return gtsam.Pose3(gtsam_R, gtsam_t)

    @classmethod
    def _init_stereo_params(cls):
        # Using the Camera class attributes, creates a gtsam.Cal3_S2Stereo object
        if cls._StereoParams is not None:
            return True
        Camera.init_class_attributes()
        K = Camera.K()
        fx, fy = K[0][0], K[1][1]
        cx, cy = K[0][2], K[1][2]
        skew = K[0][1]
        baseline = - Camera._RightTranslation[0][0]
        gtsam_stereo_params = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, baseline)
        cls._StereoParams = gtsam_stereo_params
        return True

