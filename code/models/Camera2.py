from __future__ import annotations
from typing import Tuple
import numpy as np

import config as c
import utils as u


class Camera2:

    _K: np.ndarray = np.array([])
    _RightRotation: np.ndarray = np.array([])
    _RightTranslation: np.ndarray = np.array([])

    def __init__(self, extrinsic_mat: np.ndarray):
        if Camera2._K.size == 0 or Camera2._RightRotation.size == 0 or Camera2._RightTranslation.size == 0:
            Camera2._read_class_attributes()
        self._extrinsic_matrix = self.__verify_matrix(extrinsic_mat, 3, 4, "Extrinsic")

    @classmethod
    def get_initial_cameras(cls) -> Tuple[Camera2, Camera2]:
        left_rot, left_trans = np.eye(3), np.zeros((3, 1))
        left_cam = Camera2(np.hstack([left_rot, left_trans]))
        right_cam = left_cam.calculate_right_camera()
        return left_cam, right_cam

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return self._K

    @property
    def right_rotation(self):
        return self._RightRotation

    @property
    def right_translation(self):
        return self._RightTranslation

    @property
    def extrinsic_matrix(self) -> np.ndarray:
        return self._extrinsic_matrix

    @property
    def projection_matrix(self) -> np.ndarray:
        return self.__verify_matrix(self._K @ self._extrinsic_matrix, 3, 4, "Projection Matrix")

    @property
    def rotation_matrix(self) -> np.ndarray:
        ext = self._extrinsic_matrix
        return self.__verify_matrix(ext[:, :3], 3, 3, "Rotation")

    @property
    def translation_vector(self) -> np.ndarray:
        ext = self._extrinsic_matrix
        return self.__verify_vector(ext[:, 3:], 3, "Translation")

    def calculate_right_camera(self) -> Camera2:
        left_rot = self.rotation_matrix
        left_trans = self.translation_vector
        right_rot = self.__verify_matrix(self._RightRotation @ left_rot, 3, 3, "Rotation")
        right_trans = self.__verify_vector(self._RightRotation @ left_trans + self._RightTranslation, 3, "Translation")
        right_ext_mat = self.__verify_matrix(np.hstack([right_rot, right_trans]), 3, 4, "Extrinsic")
        return Camera2(right_ext_mat)

    def project(self, landmarks: np.ndarray) -> np.ndarray:

        # verify shape is (3, N)
        assert landmarks.shape[0] == 3 or landmarks.shape[1] == 3, "landmarks must contain 3D coordinates"
        if landmarks.shape[0] != 3:
            landmarks = landmarks.T

        K = self.intrinsic_matrix
        R = self.rotation_matrix
        t = self.translation_vector
        projections_3d = K @ (R @ landmarks + t)  # non normalized homogeneous coordinates of shape 3xN
        hom_coordinates = projections_3d / (projections_3d[2] + c.Epsilon)  # add epsilon to avoid 0 division
        return hom_coordinates[:2]  # return only first 2 rows (x,y coordinates)

    @classmethod
    def _read_class_attributes(cls):
        K, Mleft, Mright = u.read_first_camera_matrices()
        Camera2._K = Camera2.__verify_matrix(K, 3, 3, "K")
        Camera2._RightRotation = Camera2.__verify_matrix(Mright[:, :3], 3, 3, "Right Rotation")
        Camera2._RightTranslation = Camera2.__verify_vector(Mright[:, 3:], 3, "Right Translation")

    @staticmethod
    def __verify_matrix(mat: np.ndarray, num_rows: int, num_cols: int, matrix_name: str):
        if not isinstance(mat, np.ndarray):
            raise TypeError(f"{matrix_name} Matrix should be a numpy array")
        if mat.shape != (num_rows, num_cols):
            raise ValueError(f"{matrix_name} Matrix shape should be {(num_rows, num_cols)}, not {mat.shape}")
        return mat

    @staticmethod
    def __verify_vector(vec: np.ndarray, length: int, vec_name: str):
        if not isinstance(vec, np.ndarray):
            raise TypeError(f"{vec_name} Vector should be a numpy array")
        if len(vec) != length:
            raise ValueError(f"{vec_name} Vector shape should be of length {length} , not {len(vec)}")
        return vec.reshape((length, 1))


