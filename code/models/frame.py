from typing import Optional

from models.match import FrameMatch
from models.camera import Camera
from models.track import Track


class Frame:
    """
    Represents a stereo rectified image-pair from the KITTI dataset
    """

    MaxIndex = 3449

    def __init__(self, idx: int, left_cam: Optional[Camera] = None, right_cam: Optional[Camera] = None):
        if idx < 0:
            raise IndexError(f"Frame index must be between 0 and {self.MaxIndex}, not {idx}")
        self.id = idx
        self._tracks = dict[Track, FrameMatch]()
        self.left_camera = left_cam
        self.right_camera = right_cam

    def get_id(self) -> int:
        return self.id

    def get_tracks(self) -> dict[Track, FrameMatch]:
        return self._tracks

    def add_track(self, track: Track, match: FrameMatch) -> bool:
        """
        Adds a new (Track, FrameMatch) pair to this Frame's internal dict.

        @throws AssertionError if the Track or FrameMatch already exist in the dict and not associated to each other.
        Returns True otherwise.
        """
        if track in self._tracks.keys():
            other_match = self._tracks[track]
            if match == other_match:
                return True
            else:
                raise AssertionError(f"{str(track)} already associated with a different FrameMatch")
        self._tracks[track] = match
        # TODO: if we require cross-check when matching, we should assert uniqueness for Match <-> Track pairings:
        # other_track = self.find_track_for_match(match)
        # if other_track == track:
        #     return True
        # if other_track is None:
        #     self._tracks[track] = match
        #     return True
        # raise AssertionError(f"FrameMatch already associated with a different Track")

    def find_track_for_match(self, match: FrameMatch) -> Optional[Track]:
        for tr, other_match in self._tracks.items():
            if match == other_match:
                return tr
        return None

    def __str__(self):
        return f"Frame{self.id}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Frame):
            return False
        get_id_attr = getattr(other, 'get_id', None)
        if get_id_attr is None:
            return False
        if not callable(get_id_attr):
            return False
        return self.id == other.get_id()
