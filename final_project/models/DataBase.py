import os
import pandas as pd
import numpy as np
import pickle as pkl
from datetime import datetime
from typing import List, Tuple

import final_project.config as c
from final_project.models.Frame import Frame
from final_project.models.Camera import Camera


class DataBase:

    def __init__(self, frames: List[Frame]):
        self._tracks_db = self.__build_tracks_database(frames)
        self._cameras_db = self.__build_cameras_database(frames)

    def get_camera(self, frame_idx: int) -> Camera:
        return self._cameras_db[frame_idx]

    def get_track_idxs(self, frame_idx: int) -> pd.Series:
        return pd.Series(self._tracks_db.xs(frame_idx, level=c.FrameIdx).index)

    def get_frame_idxs(self, track_idx: int) -> pd.Series:
        return pd.Series(self._tracks_db.xs(track_idx, level=c.TrackIdx).index)

    def get_coordinates(self, frame_idx: int, track_idx: int) -> np.ndarray:
        return np.array(self._tracks_db.loc[(frame_idx, track_idx), [c.XL, c.XR, c.Y]])

    def get_track_lengths(self) -> pd.Series:
        return self._tracks_db.groupby(level=c.TrackIdx).size()

    def prune_short_tracks(self, min_length: int, inplace=False):
        assert min_length >= 1, "Track length must be a positive integer"
        track_lengths = self.get_track_lengths()
        short_track_idxs = track_lengths[track_lengths < min_length].index
        new_tracks_db = self._tracks_db.drop(short_track_idxs, level=c.TrackIdx)
        if inplace:
            self._tracks_db = new_tracks_db
            return None
        db = DataBase(frames=[])
        db._tracks_db = new_tracks_db
        db._cameras_db = self._cameras_db
        return db

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
        idx1_tracks = self._tracks_db.index[self._tracks_db.index.get_level_values(c.FrameIdx) == frame_idx1].droplevel(c.FrameIdx)
        idx2_tracks = self._tracks_db.index[self._tracks_db.index.get_level_values(c.FrameIdx) == frame_idx2].droplevel(c.FrameIdx)
        return idx1_tracks.intersection(idx2_tracks).to_series().reset_index(drop=True)

    def to_pickle(self) -> bool:
        if not os.path.isdir(c.DATA_WRITE_PATH):
            os.makedirs(c.DATA_WRITE_PATH)
        filename_base = f"db_{datetime.now().strftime('%d%m%Y_%H%M')}"
        tracks_filename = "tracks" + filename_base
        cameras_filename = "cameras" + filename_base
        with open(os.path.join(c.DATA_WRITE_PATH, tracks_filename + ".pkl"), 'wb') as f:
            pkl.dump(self._tracks_db, f, protocol=-1)
        with open(os.path.join(c.DATA_WRITE_PATH, cameras_filename + ".pkl"), 'wb') as f:
            pkl.dump(self._cameras_db, f, protocol=-1)
        return True

    @staticmethod
    def from_pickle(tracks_file: str, cameras_file: str):
        db = DataBase(frames=[])
        db._tracks_db = DataBase.__read_pickle(tracks_file)
        db._cameras_db = DataBase.__read_pickle(cameras_file)
        Camera.read_initial_cameras()  # initiates the private class attributes of class Camera
        return db

    @staticmethod
    def __build_tracks_database(frames: List[Frame]):
        if len(frames) == 0:
            tracks_db = pd.DataFrame(columns=[c.XL, c.XR, c.Y, c.FrameIdx, c.TrackIdx])
            tracks_db.set_index([c.FrameIdx, c.TrackIdx], inplace=True)
            return tracks_db

        mini_dbs = {}
        for fr in frames:
            mini_dbs[fr.idx] = fr.extract_all_tracks()
        tracks_db = pd.concat(mini_dbs)
        tracks_db.index.set_names([c.FrameIdx, c.TrackIdx], inplace=True)
        return tracks_db

    @staticmethod
    def __build_cameras_database(frames: List[Frame]):
        cams = pd.Series([fr.left_cam for fr in frames], name=c.CamL)
        cams.index.name = c.FrameIdx
        return cams

    @staticmethod
    def __read_pickle(p: str):
        if not p.endswith(".pkl"):
            p += ".pkl"
        if os.path.dirname(p) != "":
            return pd.read_pickle(p)
        fullpath = os.path.join(c.DATA_WRITE_PATH, p)
        return pd.read_pickle(fullpath)


