import numpy as np

import final_project.config as c
import final_project.utils as u


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
        _, M_left, M_right = u.read_first_camera_matrices()
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

    @classmethod
    def __init_class_attributes(cls) -> bool:
        if cls._is_init:
            return True
        try:
            cls.__verify_shape(cls._K, 3, 3, "Intrinsic")
            cls.__verify_shape(cls._RightRotation, 3, 3, "Right Rotation")
            cls.__verify_shape(cls._RightTranslation, 3, 1, "Right Translation")
        except AssertionError:
            K, _, M_right = u.read_first_camera_matrices()
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

