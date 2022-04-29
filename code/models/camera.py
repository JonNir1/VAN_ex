import os
import numpy as np

import config as c
from models.directions import Side


class Camera:
    """
    Each Camera object is either the Left or Right camera in a stereo-rectified camera-pair.
    A camera has its intrinsic 3x3 matrix (K), and a 3x4 extrinsic matrix ([R|t]),
    comprised of a 3x3 rotation matrix (R) and a 3x1 translation vector (t)
    """

    @staticmethod
    def read_first_cameras():
        """
        Load camera matrices from the KITTY dataset
        Returns 2 Camera objects with the following matrices:
            K - Intrinsic camera matrix
            M_left, M_right - Extrinsic camera matrix (left, right)
        """
        with open(os.path.join(c.DATA_READ_PATH, 'calib.txt'), "r") as f:
            l1 = f.readline().split()[1:]  # skip first token
            l2 = f.readline().split()[1:]  # skip first token
        l1 = [float(i) for i in l1]
        m1 = np.array(l1).reshape(3, 4)
        l2 = [float(i) for i in l2]
        m2 = np.array(l2).reshape(3, 4)
        K = m1[:, :3]
        M_left = np.linalg.inv(K) @ m1
        M_right = np.linalg.inv(K) @ m2
        left_cam = Camera(0, Side.LEFT, K, M_left)
        right_cam = Camera(0, Side.RIGHT, K, M_right)
        return left_cam, right_cam

    def __init__(self, idx: int, side: Side,
                 intrinsic_mat: np.ndarray, extrinsic_mat: np.ndarray):
        self.idx = idx
        self.side = side
        self._intrinsic_matrix = self.__verify_matrix(intrinsic_mat, 3, 3, "Intrinsic")
        self._extrinsic_matrix = self.__verify_matrix(extrinsic_mat, 3, 4, "Extrinsic")

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return self._intrinsic_matrix

    @property
    def extrinsic_matrix(self) -> np.ndarray:
        return self._extrinsic_matrix

    def project_3d_points(self, points: np.ndarray) -> np.ndarray:
        # returns a 2xN array of the projected points onto the Camera's plane
        assert points.shape[0] == 3 or points.shape[1] == 3, \
            f"Must provide a 3D points matrix, input has shape {points.shape}"
        if points.shape[0] != 3:
            points = points.T

        K = self._intrinsic_matrix
        R = self.get_rotation_matrix()
        t = self.get_translation_vector()
        projections = K @ (R @ points + t)  # non normalized homogeneous coordinates of shape 3xN
        hom_coordinates = projections / (projections[2] + c.Epsilon)  # add epsilon to avoid 0 division
        return hom_coordinates[:2]  # return only first 2 rows (x,y coordinates)

    def calculate_projection_matrix(self) -> np.ndarray:
        # returns a 3x4 ndarray that maps a 3D point ro its corresponding 2D point on the camera plain
        return self._intrinsic_matrix @ self._extrinsic_matrix

    def get_rotation_matrix(self) -> np.ndarray:
        r = self._extrinsic_matrix[:, :3]
        return self.__verify_matrix(r, 3, 3, "Rotation")

    def get_translation_vector(self) -> np.ndarray:
        t = self._extrinsic_matrix[:, 3:]
        return self.__verify_vector(t, 3, "Translation")

    def calculate_right_camera(self):
        """
        Calculates the extrinsic matrix of the right Camera, based on $self, and the first right Camera
        Returns a Camera object if successful
        """
        _, first_right_cam = Camera.read_first_cameras()
        right_rot0 = first_right_cam.get_rotation_matrix()
        right_trans0 = first_right_cam.get_translation_vector()
        front_left_rot = self.get_rotation_matrix()
        front_left_trans = self.get_translation_vector()

        front_right_Rot = right_rot0 @ front_left_rot
        front_right_trans = right_rot0 @ front_left_trans + right_trans0
        ext_mat = Camera.calculate_extrinsic_matrix(front_right_Rot, front_right_trans)
        return Camera(idx=self.idx, side=Side.RIGHT, intrinsic_mat=self.intrinsic_matrix, extrinsic_mat=ext_mat)

    @staticmethod
    def calculate_extrinsic_matrix(rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
        r = Camera.__verify_matrix(rotation_matrix, 3, 3, "Rotation")
        t = Camera.__verify_vector(translation_vector, 3, "Translation")
        return np.hstack([r, t])

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

    def __str__(self):
        return f"Cam_{self.idx}{self.side.value}"

