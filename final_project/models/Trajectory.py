import numpy as np
from typing import Iterable

import final_project.config as c
import final_project.logic.Utils as u
from final_project.logic.Utils import Axis
from final_project.models.Camera import Camera


class Trajectory:

    def __init__(self, coords: np.ndarray):
        if coords.shape[0] != 3:
            coords = coords.T
        dims, length = coords.shape
        assert dims == 3, f"Trajectory needs to have 3 rows (X, Y, Z coordinates)"
        self._length = length
        self.coordinates = coords

    @staticmethod
    def from_ground_truth(num_frames: int = c.NUM_FRAMES):
        """
        Reads KITTI's ground-truth Cameras and calculates the trajectories from them
        """
        abs_cameras = u.read_ground_truth_cameras(use_relative=False)
        coords = np.array([cam.calculate_coordinates() for cam in abs_cameras])
        return Trajectory(coords.T)

    @staticmethod
    def from_absolute_cameras(cameras: Iterable[Camera]):
        coords_list = [cam.calculate_coordinates() for cam in cameras]
        coords = np.array(coords_list)
        return Trajectory(coords)

    @staticmethod
    def from_relative_cameras(cameras: Iterable[Camera]):
        absolute_cameras = u.convert_to_absolute_cameras(cameras)
        return Trajectory.from_absolute_cameras(absolute_cameras)

    @property
    def X(self) -> np.ndarray:
        return self._get_axis(Axis.X)

    @property
    def Y(self) -> np.ndarray:
        return self._get_axis(Axis.Y)

    @property
    def Z(self) -> np.ndarray:
        return self._get_axis(Axis.Z)

    def calculate_distance(self, other, axis: Axis = Axis.ALL) -> np.ndarray:
        if not isinstance(other, Trajectory):
            raise TypeError("other must be of type Trajectory")
        self_coords = self._get_axis(axis)
        other_coords = other._get_axis(axis)
        return np.linalg.norm(self_coords - other_coords, ord=2, axis=0)

    def _get_axis(self, axis: Axis) -> np.ndarray:
        if axis == Axis.X:
            return self.coordinates[0]
        if axis == Axis.Y:
            return self.coordinates[1]
        if axis == Axis.Z:
            return self.coordinates[2]
        if axis == Axis.ALL:
            return self.coordinates
        raise TypeError("axis must be of type Axis")

    def __len__(self):
        return self._length

    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            return False
        if len(self) != len(other):
            return False
        return (self.coordinates == other.coordinates).all().all()

    def __getitem__(self, idx) -> np.ndarray:
        if not isinstance(idx, int):
            # TODO: support slicing (e.g. [:3])
            # https://medium.com/@beef_and_rice/mastering-pythons-getitem-and-slicing-c94f85415e1c
            raise TypeError("Trajectory indexers must be int")
        max_index = len(self) - 1
        if idx < 0 or idx > max_index:
            raise IndexError(f"Index {idx} is out of bounds (0 to {max_index})")
        return (self.coordinates[:, idx]).reshape((3,))
