import numpy as np

from models.directions import Side


class Camera:
    """
    Each Camera object is either the Left or Right camera in a stereo-rectified camera-pair.
    A camera has its intrinsic 3x3 matrix (K), and a 3x4 extrinsic matrix ([R|t]),
    comprised of a 3x3 rotation matrix (R) and a 3x1 translation vector (t)
    """

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

    @extrinsic_matrix.setter
    def extrinsic_matrix(self, rotation_matrix: np.ndarray, translation_vector: np.ndarray):
        r = self.__verify_matrix(rotation_matrix, 3, 3, "Rotation")
        t = self.__verify_vector(translation_vector, 3, "Translation")
        self._extrinsic_matrix = np.hstack([r, t])

    def calculate_projection_matrix(self) -> np.ndarray:
        # returns a 3x4 ndarray that maps a 3D point ro its corresponding 2D point on the camera plain
        return self._intrinsic_matrix @ self._extrinsic_matrix

    def get_rotation_matrix(self) -> np.ndarray:
        r = self._extrinsic_matrix[:, :3]
        return self.__verify_matrix(r, 3, 3, "Rotation")

    def get_translation_vector(self) -> np.ndarray:
        t = self._extrinsic_matrix[:, 3:]
        return self.__verify_vector(t, 3, "Translation")

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
            raise TypeError(f"{vec_name} Matrix should be a numpy array")
        if vec.shape != (length,) or vec.shape != (length, 1):
            raise ValueError(f"{vec_name} Matrix shape should be {(length, 1)}, not {vec.shape}")
        return vec.reshape((length, 1))

    def __str__(self):
        return f"Cam_{self.idx}{self.side.value}"

