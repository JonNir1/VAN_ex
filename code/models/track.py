from itertools import count


class Track:
    """
    A 3D landmark that was identified in the KITTI dataset.
    Each Track instance has its own unique ID, and is associated with all the frames and keypoints that depict it.
    """
    __ids = count(0)

    def __init__(self, start_frame_idx: int):
        self._id = next(self.__ids)
        self._start_frame_id = start_frame_idx
        self._end_frame_id = start_frame_idx

    def get_id(self):
        return self._id

    def get_length(self) -> int:
        return 1 + self._end_frame_id - self._start_frame_id

    def get_frame_ids(self) -> list[int]:
        return [idx for idx in range(self._start_frame_id, self._end_frame_id + 1)]

    def extend(self):
        self._end_frame_id += 1

    def _get_start_frame_id(self):
        return self._start_frame_id

    def _get_end_frame_id(self):
        return self._end_frame_id

    def __str__(self):
        return f"Track{self._id}"

