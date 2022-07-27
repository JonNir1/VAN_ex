import pandas as pd
from typing import Optional, Dict, Tuple

import final_project.config as c
from final_project.models.Camera import Camera
from final_project.models.Matcher import Matcher, DEFAULT_MATCHER


class Frame:

    def __init__(self, idx: int, left_cam: Optional[Camera] = None, matcher: Matcher = DEFAULT_MATCHER):
        if idx < 0 or idx >= c.NUM_FRAMES:
            raise IndexError(f"Frame index must be between 0 and {c.NUM_FRAMES - 1}, not {idx}")
        self.idx: int = idx
        self.left_cam = left_cam
        self.features, self.descriptors = matcher.match_within_frame(idx)
        self._feature_idx_to_track_id: Dict[int, int] = dict()

    @property
    def num_features(self) -> int:
        if self.features is None or self.features.size == 0:
            return 0
        return self.features.shape[0]

    def get_track_id(self, feature_idx: int) -> Optional[int]:
        track_id = self._feature_idx_to_track_id.get(feature_idx)
        return track_id

    def set_track_id(self, feature_idx: int, track_id: int):
        self._feature_idx_to_track_id[feature_idx] = track_id

    def extract_all_tracks(self) -> pd.DataFrame:
        tracks_data = dict()
        for feature_idx, track_idx in self._feature_idx_to_track_id.items():
            xl, yl, xr, yr = self.features[feature_idx]
            tracks_data[track_idx] = xl, xr, yl
        df = pd.DataFrame(tracks_data).T
        df.rename(columns={0: c.XL, 1: c.XR, 2: c.Y}, inplace=True)
        df.index.name = c.TrackIdx
        df.sort_index(inplace=True)
        return df

    def __str__(self):
        return f"Fr{self.idx}"

    def __eq__(self, other):
        if not isinstance(other, Frame):
            return False
        if not self.idx == other.idx:
            return False
        return self.num_features == other.num_features

