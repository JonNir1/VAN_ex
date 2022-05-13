import os
import numpy as np
import pandas as pd
import pickle as pkl
from datetime import datetime
from typing import List, Tuple

import config as c
from models.database import DataBase
from models.frame import Frame


class DBAdapter:
    # This class is used for accessing the DataBase without exposing any "write" privileges on the DB

    def __init__(self, data: List[Frame]):
        self.tracks_db = DataBase.build_tracks_database(data)
        self.cameras_db = DataBase.build_cameras_database(data)

    @staticmethod
    def from_pickles(tracks_file: str, cameras_file: str):
        tracks_db = DataBase.from_pickle(tracks_file)
        cameras_db = DataBase.from_pickle(cameras_file)
        # TODO: assert correct structure
        dba = DBAdapter(data=[])
        dba.tracks_db = tracks_db
        dba.cameras_db = cameras_db
        return dba

    def get_track_idxs(self, frame_idx: int) -> pd.Series:
        return pd.Series(self.tracks_db.xs(frame_idx, level=DataBase.FRAMEIDX).index)

    def get_frame_idxs(self, track_idx: int) -> pd.Series:
        return pd.Series(self.tracks_db.xs(track_idx, level=DataBase.TRACKIDX).index)

    def get_track_lengths(self) -> pd.Series:
        return self.tracks_db.groupby(level=DataBase.TRACKIDX).size()

    def sample_track_idx_with_length(self, min_len: int = 10, max_len: int = None) -> Tuple[int, int]:
        track_lengths = self.get_track_lengths()
        if max_len is None:
            relevant_indices = track_lengths >= min_len
        else:
            assert max_len >= min_len, f"argument $max_len ({max_len}) must be >= to argument $min_len ({min_len})"
            relevant_indices = (track_lengths >= min_len) & (track_lengths <= max_len)
        track_with_len = (track_lengths[relevant_indices]).sample(1)
        trk_idx = track_with_len.index.values[0]
        length = track_with_len.values[0]
        return trk_idx, length

    def get_shared_tracks(self, frame_idx1: int, frame_idx2: int) -> pd.Series:
        idx1_tracks = self.tracks_db.index[self.tracks_db.index.get_level_values(DataBase.FRAMEIDX) == frame_idx1].droplevel(DataBase.FRAMEIDX)
        idx2_tracks = self.tracks_db.index[self.tracks_db.index.get_level_values(DataBase.FRAMEIDX) == frame_idx2].droplevel(DataBase.FRAMEIDX)
        return idx1_tracks.intersection(idx2_tracks).to_series().reset_index(drop=True)

    def get_coordinates(self, frame_idx: int, track_idx: int) -> np.ndarray:
        return np.array(self.tracks_db.loc[(frame_idx, track_idx),
                                           [DataBase.X_LEFT, DataBase.X_RIGHT, DataBase.Y]])

    def to_pickle(self) -> bool:
        if not os.path.isdir(c.DATA_WRITE_PATH):
            os.makedirs(c.DATA_WRITE_PATH)
        filename_base = f"db_{datetime.now().strftime('%d%m%Y_%H%M')}"
        tracks_filename = "tracks" + filename_base
        cameras_filename = "cameras" + filename_base
        with open(os.path.join(c.DATA_WRITE_PATH, tracks_filename + ".pkl"), 'wb') as f:
            pkl.dump(self.tracks_db, f, protocol=-1)
        with open(os.path.join(c.DATA_WRITE_PATH, cameras_filename + ".pkl"), 'wb') as f:
            pkl.dump(self.cameras_db, f, protocol=-1)
        return True

    def __eq__(self, other):
        if not isinstance(other, DBAdapter):
            return False
        return (self.tracks_db == other.tracks_db).all().all()



