from typing import NamedTuple, Tuple

import numpy as np


class MatchedFeature(NamedTuple):
    Xl: float  # left X coordinate
    Yl: float  # left Y coordinate
    Xr: float  # right X coordinate
    Yr: float  # right Y coordinate

    def get_left(self) -> Tuple[float, float]:
        return self.Xl, self.Yl

    def get_right(self) -> Tuple[float, float]:
        return self.Xr, self.Yr

    def get_pixels(self) -> np.ndarray:
        # returns a 2x2 array where rows are (left, right) and cols are (X, Y)
        left = np.array(self.get_left())
        right = np.array(self.get_right())
        return np.vstack([left, right])

    def __str__(self):
        return f"(({self.Xl:.3f}, {self.Yl:.3f}),\t({self.Xr:.3f}, {self.Yr:.3f}))"

    def __eq__(self, other):
        if not isinstance(other, MatchedFeature):
            return False
        if self.Xl != other.Xl or self.Yl != other.Yl:
            return False
        if self.Xr != other.Xr or self.Yr != other.Yr:
            return False
        return True

