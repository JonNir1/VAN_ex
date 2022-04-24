from itertools import count

from models.frame import Frame


class Track:
    """
    A 3D landmark that was identified in the KITTI dataset.
    Each Track instance has its own unique ID, and is associated with all the frames and keypoints that depict it.
    """
    __ids = count(0)

    def __init__(self, start_frame: Frame):
        self._idx = next(self.__ids)
        self._frames = {start_frame}

    def get_idx(self):
        return self._idx

    def get_frames(self):
        return self._frames

    def get_frame_ids(self):
        return [f.get_idx() for f in self._frames]

    def get_length(self):
        return len(self._frames)

    def add_frame(self, new_frame: Frame):
        self._frames.add(new_frame)

    def __str__(self):
        return f"Track{self._idx}"

