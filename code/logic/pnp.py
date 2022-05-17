import cv2
import numpy as np
from typing import Optional, List, Tuple

from models.camera import Camera
from models.match import MutualMatch
from models.directions import Side, Position
from logic.triangulation import triangulate_matches


MinSamplesNumber = 4


def compute_front_cameras(mutual_matches: List[MutualMatch], bl_cam: Camera, br_cam: Camera
                          ) -> Tuple[Optional[Camera], Optional[Camera]]:
    """
    Calculate the two front Camera objects, Based on multiple MutualMatch objects (4-way match of keypoints) and
        the back pair of Camera objects.
    Uses cv2.solvePnP to compute the front-left Camera, and then uses that result to compute the front-right Camera.

    :raises: AssertionError if there are less than 4 MutualMatch to base the calculation on.
    :returns: a pair of Cameras if the calculation was successful, or a pair of None objects otherwise
    """
    assert len(mutual_matches) >= MinSamplesNumber, f"minimum number of matches for PnP is 4"
    fl_cam = _compute_front_left_camera(mutual_matches, bl_cam, br_cam)
    if fl_cam is None:
        return None, None
    fr_cam = fl_cam.calculate_right_camera()
    return fl_cam, fr_cam


def _compute_front_left_camera(mutual_matches: List[MutualMatch], bl_cam: Camera, br_cam: Camera) -> Optional[Camera]:
    # TODO: perform PnP with RELATIVE cameras (back_left_camera @ coordinate (0,0,0))
    next_idx = bl_cam.idx + 1
    K = bl_cam.intrinsic_matrix
    back_frame_matches = [m.get_frame_match(Position.BACK) for m in mutual_matches]
    point_cloud_3d = (triangulate_matches(back_frame_matches, bl_cam, br_cam)).T                            # shape (3, N)
    front_left_pixels = np.array([(m.get_keypoint(Side.LEFT, Position.FRONT)).pt for m in mutual_matches])  # shape (N, 2)
    success, rotation, translation = cv2.solvePnP(objectPoints=point_cloud_3d, imagePoints=front_left_pixels,
                                                  cameraMatrix=K, distCoeffs=None, flags=cv2.SOLVEPNP_EPNP)
    if success:
        ext_mat = Camera.calculate_extrinsic_matrix(cv2.Rodrigues(rotation)[0], translation)
        return Camera(idx=next_idx, side=Side.LEFT, extrinsic_mat=ext_mat)
    return None

