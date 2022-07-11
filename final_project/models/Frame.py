import numpy as np
from typing import Optional

import final_project.config as c
from final_project.models.Camera import Camera
from final_project.models.Matcher import Matcher


class Frame:

    def __init__(self, idx: int, left_cam: Optional[Camera] = None, matcher: Matcher = c.DEFAULT_MATCHER):
        if idx < 0 or idx >= c.NUM_FRAMES:
            raise IndexError(f"Frame index must be between 0 and {c.NUM_FRAMES - 1}, not {idx}")
        self.idx: int = idx
        self.left_cam = left_cam
        self.features, self.descriptors = matcher.match_within_frame(idx)

    @property
    def num_features(self) -> int:
        if self.features is None or self.features.size == 0:
            return 0
        return self.features.shape[0]

    def __str__(self):
        return f"Fr{self.idx}"

    def __eq__(self, other):
        if not isinstance(other, Frame):
            return False
        if not self.idx == other.idx:
            return False
        return self.num_features == other.num_features

