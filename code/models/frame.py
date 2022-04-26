from typing import Optional

import config as c
from models.match import FrameMatch
from models.camera import Camera
from models.track import Track


class Frame:
    """
    Represents a stereo rectified image-pair from the KITTI dataset
    """

    MaxIndex = 3449

    def __init__(self, idx: int):
        if idx < 0:
            raise IndexError(f"Frame index must be between 0 and {self.MaxIndex}, not {idx}")
        self.id = idx
        self._tracks = dict[Track, FrameMatch]()  # a dict matching Track to (keypoint_left, keypoint_right)
        self._left_camera = None
        self._right_camera = None

    def get_id(self) -> int:
        return self.id

    def get_tracks(self) -> dict[Track, FrameMatch]:
        return self._tracks

    @property
    def left_camera(self) -> Optional[Camera]:
        return self._left_camera

    @left_camera.setter
    def left_camera(self, cam: Optional[Camera]):
        self._left_camera = cam

    @property
    def right_camera(self) -> Optional[Camera]:
        return self._right_camera

    @right_camera.setter
    def right_camera(self, cam: Optional[Camera]):
        self._right_camera = cam

    def get_track_for_point(self, x: float, y: float) -> Optional[Track]:
        """
        Returns a Track object that has a projection on this frame's left image that is
        identical to the provided (x,y) coordinates, or None if no such track exists.
        """
        for t, coords in self._tracks.items():
            x_left, x_right, y_left = coords
            if abs(x_left - x) <= c.Epsilon and abs(y_left - y) <= c.Epsilon:
                return t
        return None

    def __str__(self):
        return f"Frame{self.id}"

