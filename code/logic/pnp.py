import cv2
import numpy as np
from typing import Optional

from models.camera import Camera
from models.match import MutualMatch
from models.directions import Side, Position
from logic.triangulation import triangulate


MinSamplesNumber = 4


def compute_front_cameras(mutual_matches: list[MutualMatch], bl_cam: Camera, br_cam: Camera
                          ) -> tuple[Optional[Camera], Optional[Camera]]:
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
    fr_cam = _compute_front_right_camera(fl_cam)
    return fl_cam, fr_cam


def _compute_front_left_camera(mutual_matches: list[MutualMatch], bl_cam: Camera, br_cam: Camera) -> Optional[Camera]:
    next_idx = bl_cam.idx + 1
    K = bl_cam.intrinsic_matrix
    back_frame_matches = [m.get_frame_match(Position.BACK) for m in mutual_matches]
    point_cloud_3d = (triangulate(back_frame_matches, bl_cam, br_cam)).T                                    # shape (3, N)
    front_left_pixels = np.array([(m.get_keypoint(Side.LEFT, Position.FRONT)).pt for m in mutual_matches])  # shape (N, 2)
    success, rotation, translation = cv2.solvePnP(objectPoints=point_cloud_3d, imagePoints=front_left_pixels,
                                                  cameraMatrix=K, distCoeffs=None, flags=cv2.SOLVEPNP_EPNP)
    if success:
        ext_mat = Camera.calculate_extrinsic_matrix(cv2.Rodrigues(rotation)[0], translation)
        return Camera(idx=next_idx, side=Side.LEFT, intrinsic_mat=K, extrinsic_mat=ext_mat)
    return None


def _compute_front_right_camera(fl_cam: Camera) -> Camera:
    _, first_right_cam = Camera.read_first_cameras()
    right_rot0 = first_right_cam.get_rotation_matrix()
    right_trans0 = first_right_cam.get_translation_vector()
    front_left_rot = fl_cam.get_rotation_matrix()
    front_left_trans = fl_cam.get_translation_vector()

    front_right_Rot = right_rot0 @ front_left_rot
    front_right_trans = right_rot0 @ front_left_trans + right_trans0
    ext_mat = Camera.calculate_extrinsic_matrix(front_right_Rot, front_right_trans)
    return Camera(idx=fl_cam.idx, side=Side.RIGHT, intrinsic_mat=fl_cam.intrinsic_matrix, extrinsic_mat=ext_mat)

