import numpy as np
from typing import NamedTuple


class Feature(NamedTuple):
    xl: float
    yl: float
    xr: float
    yr: float
    desc: np.ndarray

    def __str__(self):
        return f"({self.xl:.2f}, {self.yl:.2f})\t({self.xr:.2f}, {self.yr:.2f})"

    def __eq__(self, other):
        if not isinstance(other, Feature):
            return False
        return self.xl == other.xl and self.yl == other.yl and self.xr == other.xr and self.yr == other.yr
