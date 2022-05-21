import gtsam
import numpy as np

import config as c
import utils as u
from models.directions import Side


class Camera:
    """
    Each Camera object is either the Left or Right camera in a stereo-rectified camera-pair.
    A camera has its intrinsic 3x3 matrix (K), and a 3x4 extrinsic matrix ([R|t]),
    comprised of a 3x3 rotation matrix (R) and a 3x1 translation vector (t)
    """

    _K: np.ndarray = np.array([])
    _RightRotation: np.ndarray = np.array([])
    _RightTranslation: np.ndarray = np.array([])

    def __init__(self, idx: int, side: Side, extrinsic_mat: np.ndarray):
        self.idx = idx
        self.side = side
        self._extrinsic_matrix = self.__verify_matrix(extrinsic_mat, 3, 4, "Extrinsic")
        if self._K.size == 0:
            self._update_class_attributes()

    @staticmethod
    def from_pose3(idx: int, pose: gtsam.Pose3):
        """
        Returns a (left) Camera object based on the provided gtsam.Pose3 object
        From a given [R|t] Pose3 matrix, the transformation to Camera matrix [R^*|t^*] is
            R^* = R.T
            t^* = - R.T @ t
        """
        R = pose.rotation().matrix().T
        t = -R @ pose.translation().reshape((3, 1))
        ext = np.hstack([R, t])
        return Camera(idx=idx, side=Side.LEFT, extrinsic_mat=ext)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return self._K

    @property
    def extrinsic_matrix(self) -> np.ndarray:
        return self._extrinsic_matrix

    @property
    def right_translation(self):
        return self._RightTranslation

    def project_3d_points(self, points: np.ndarray) -> np.ndarray:
        # returns a 2xN array of the projected points onto the Camera's plane
        assert points.shape[0] == 3 or points.shape[1] == 3, \
            f"Must provide a 3D points matrix, input has shape {points.shape}"
        if points.shape[0] != 3:
            points = points.T

        R = self.get_rotation_matrix()
        t = self.get_translation_vector()
        projections = self._K @ (R @ points + t)  # non normalized homogeneous coordinates of shape 3xN
        hom_coordinates = projections / (projections[2] + c.Epsilon)  # add epsilon to avoid 0 division
        return hom_coordinates[:2]  # return only first 2 rows (x,y coordinates)

    def calculate_projection_matrix(self) -> np.ndarray:
        # returns a 3x4 ndarray that maps a 3D point ro its corresponding 2D point on the camera plain
        return self._K @ self._extrinsic_matrix

    def calculate_coordinates(self) -> np.ndarray:
        # Returns the 3D coordinates of the Camera in the GLOBAL (first Frame's) coordinate system
        R = self.get_rotation_matrix()
        t = self.get_translation_vector()
        global_position = - (R.T @ t).reshape((3,))
        return global_position

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
        front_left_rot = self.get_rotation_matrix()
        front_left_trans = self.get_translation_vector()
        front_right_Rot = self._RightRotation @ front_left_rot
        front_right_trans = self._RightRotation @ front_left_trans + self._RightTranslation
        ext_mat = Camera.calculate_extrinsic_matrix(front_right_Rot, front_right_trans)
        return Camera(idx=self.idx, side=Side.RIGHT, extrinsic_mat=ext_mat)

    @staticmethod
    def calculate_extrinsic_matrix(rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
        r = Camera.__verify_matrix(rotation_matrix, 3, 3, "Rotation")
        t = Camera.__verify_vector(translation_vector, 3, "Translation")
        return np.hstack([r, t])

    @classmethod
    def _update_class_attributes(cls):
        K, Mleft, Mright = u.read_first_camera_matrices()
        Camera._K = Camera.__verify_matrix(K, 3, 3, "K")
        Camera._RightRotation = Camera.__verify_matrix(Mright[:, :3], 3, 3, "Right Rotation")
        Camera._RightTranslation = Camera.__verify_vector(Mright[:, 3:], 3, "Right Translation")

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
