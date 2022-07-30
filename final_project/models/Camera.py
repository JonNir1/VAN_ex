import os
import math
import numpy as np

import final_project.config as c


def read_first_camera_matrices(path: str = ""):
    """
    Load camera matrices from the KITTY dataset
    Returns the following matrices (ndarrays):
        K - Intrinsic camera matrix
        M_left, M_right - Extrinsic camera matrix (left, right)
    """
    if path is None or path == "":
        path = os.path.join(c.DATA_READ_PATH, "sequences", "00", "calib.txt")
    with open(path, "r") as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    K = m1[:, :3]
    M_left = np.linalg.inv(K) @ m1
    M_right = np.linalg.inv(K) @ m2
    return K, M_left, M_right


class Camera:

    _is_init: bool = False
    _K: np.ndarray = np.array([])
    _RightRotation: np.ndarray = np.array([])
    _RightTranslation: np.ndarray = np.array([])

    def __init__(self, extrinsic: np.ndarray):
        self.__verify_shape(extrinsic, 3, 4, "Extrinsic")
        self._M: np.ndarray = extrinsic
        Camera._is_init = Camera.init_class_attributes()

    @staticmethod
    def from_Rt(R: np.ndarray, t: np.ndarray):
        Camera.__verify_shape(R, 3, 3, "Rotation")
        Camera.__verify_shape(t, 3, 1, "Translation")
        return Camera(np.hstack([R, t]))

    @staticmethod
    def read_initial_cameras():
        _, M_left, M_right = read_first_camera_matrices()
        left_cam = Camera(M_left)
        right_cam = Camera(M_right)
        return left_cam, right_cam

    @classmethod
    def init_class_attributes(cls) -> bool:
        # TODO: delete if this is not used
        return cls.__init_class_attributes()

    @classmethod
    def K(cls) -> np.ndarray:
        return Camera._K

    @property
    def R(self) -> np.ndarray:
        return self._M[:, :-1]

    @property
    def t(self) -> np.ndarray:
        t = self._M[:, -1]
        return t.reshape((3, 1))

    @property
    def projection_matrix(self) -> np.ndarray:
        return Camera.K() @ self._M

    def calculate_coordinates(self) -> np.ndarray:
        R, t = self.R, self.t
        return (-R.T @ t).reshape((3,))

    def get_right_camera(self):
        right_rot = Camera._RightRotation @ self.R
        right_trans = Camera._RightRotation @ self.t + Camera._RightTranslation
        return Camera.from_Rt(right_rot, right_trans)

    def angles_between(self, other, use_degrees=True) -> np.ndarray:
        assert isinstance(other, Camera), "other object must be a Camera"
        relative_rotation = self.R.T @ other.R
        return Camera._rotation_to_angles(relative_rotation, use_degrees)

    @staticmethod
    def _rotation_to_angles(R: np.ndarray, use_degrees=True) -> np.ndarray:
        # Converts a rotation matrix to yaw-pitch-roll (Euler angles)
        # see https://learnopencv.com/rotation-matrix-to-euler-angles/
        assert R.shape == (3, 3), f"R should be a 3×3 matrix, but shape is {R.shape}"
        assert np.linalg.norm(R.T @ R - np.eye(3)) <= 1e-6, "R should be a Unitary matrix"
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if singular:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0
        else:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        if use_degrees:
            yaw = math.degrees(yaw)
            pitch = math.degrees(pitch)
            roll = math.degrees(roll)
        return np.array([yaw, pitch, roll])

    @classmethod
    def __init_class_attributes(cls) -> bool:
        if cls._is_init:
            return True
        try:
            cls.__verify_shape(cls._K, 3, 3, "Intrinsic")
            cls.__verify_shape(cls._RightRotation, 3, 3, "Right Rotation")
            cls.__verify_shape(cls._RightTranslation, 3, 1, "Right Translation")
        except AssertionError:
            K, _, M_right = read_first_camera_matrices()
            cls._K = K
            cls._RightRotation = M_right[:, :-1]
            cls._RightTranslation = M_right[:, -1].reshape((3, 1))
        return True

    @staticmethod
    def __verify_shape(mat: np.ndarray, nrows: int, ncols: int, name: str):
        name = name.capitalize()
        array_type = "Vector" if ncols == 1 else "Matrix"
        assert isinstance(mat, np.ndarray), f"{name} {array_type} should be a numpy array"
        assert mat.size != 0, f"Cannot provide an empty {name} {array_type}"
        assert mat.shape[0] == nrows, f"{name} {array_type}'s n_rows should be {nrows}, not {mat.shape[0]}"
        assert mat.shape[1] == ncols, f"{name} {array_type}'s n_cols should be {ncols}, not {mat.shape[1]}"

