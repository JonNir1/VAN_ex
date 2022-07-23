import os
import numpy as np
from typing import Iterable
import enum

import final_project.config as c
from final_project.models.Camera import Camera


class Axis(enum.Enum):
    X = 0
    Y = 1
    Z = 2
    All = 3


class Trajectory:

    def __init__(self, coords: np.ndarray):
        if coords.shape[0] != 3:
            coords = coords.T
        dims, length = coords.shape
        assert dims == 3, f"Trajectory needs to have 3 rows (X, Y, Z coordinates)"
        self._length = length
        self.coordinates = coords

    @staticmethod
    def read_ground_truth(num_frames: int = c.NUM_FRAMES):
        """
        Reads KITTI's ground-truth Cameras and calculates the trajectories from them
        """
        path = os.path.join(c.DATA_READ_PATH, 'poses', '00.txt')
        coords = np.zeros((num_frames, 3))
        f = open(path, 'r')
        for i, line in enumerate(f.readlines()):
            if i == num_frames:
                break
            mat = np.array(line.split(), dtype=float).reshape((3, 4))
            R, t = mat[:, :3], mat[:, 3:]
            coords[i] -= (R.T @ t).reshape((3,))
        return Trajectory(coords)

    @staticmethod
    def from_relative_cameras(cameras: Iterable[Camera]):
        abs_Rs, abs_ts = [], []
        coords_list = []
        for i, cam in enumerate(cameras):
            rel_R, rel_t = cam.R, cam.t
            if i == 0:
                curr_abs_R = rel_R
                curr_abs_t = rel_t
            else:
                prev_abs_R, prev_abs_t = abs_Rs[-1], abs_ts[-1]
                curr_abs_R = rel_R @ prev_abs_R
                curr_abs_t = rel_t.reshape((3, 1)) + (rel_R @ prev_abs_t).reshape(3, 1)
            abs_Rs.append(curr_abs_R)
            abs_ts.append(curr_abs_t)
            coords_list.append((- curr_abs_R.T @ curr_abs_t).reshape((3,)))
        coords = np.array(coords_list)
        return Trajectory(coords)

    @property
    def X(self) -> np.ndarray:
        return self._get_axis(Axis.X)

    @property
    def Y(self) -> np.ndarray:
        return self._get_axis(Axis.Y)

    @property
    def Z(self) -> np.ndarray:
        return self._get_axis(Axis.Z)

    def calculate_distance(self, other, axis: Axis = Axis.All) -> np.ndarray:
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
        if axis == Axis.All:
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
