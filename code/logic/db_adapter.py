import os
import numpy as np
import pandas as pd
import pickle as pkl
from datetime import datetime

import config as c
from models.database import DataBase
from models.frame import Frame


class DBAdapter:
    # This class is used for accessing the DataBase without exposing any "write" privileges on the DB

    def __init__(self, data: list[Frame]):
        self.db = DataBase.build_database(data)

    @staticmethod
    def from_pickle(filename: str):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        fullpath = os.path.join(c.DATA_WRITE_PATH, filename)
        df = pd.read_pickle(fullpath)
        # TODO: assert correct structure
        dba = DBAdapter(data=[])
        dba.db = df
        return dba

    def get_track_idxs(self, frame_idx: int) -> pd.Series:
        return pd.Series(self.db.xs(frame_idx, level=DataBase.FRAMEIDX).index)

    def get_frame_idxs(self, track_idx: int) -> pd.Series:
        return pd.Series(self.db.xs(track_idx, level=DataBase.TRACKIDX).index)

    def get_track_lengths(self) -> pd.Series:
        return self.db.groupby(level=DataBase.TRACKIDX).size()

    def get_track_idx_with_length(self, min_length: int) -> int:
        track_lengths = self.get_track_lengths()
        track_idx = (track_lengths[track_lengths >= min_length]).sample(1)
        return track_idx

    def get_coordinates(self, frame_idx: int, track_idx: int) -> np.ndarray:
        return np.array(self.db.loc[(frame_idx, track_idx),
                                    [DataBase.X_LEFT, DataBase.X_RIGHT, DataBase.Y]])

    def to_pickle(self, filename: str = None) -> bool:
        # TODO: can't pickle a CV2.Keypoint object
        if not os.path.isdir(c.DATA_WRITE_PATH):
            os.makedirs(c.DATA_WRITE_PATH)
        filename = filename if filename is not None else f"db_{datetime.now().strftime('%d%m%Y_%H%M')}"
        fullpath = os.path.join(c.DATA_WRITE_PATH, filename + ".pkl")
        with open(fullpath, 'wb') as f:
            pkl.dump(self.db, f, protocol=-1)
        return True



