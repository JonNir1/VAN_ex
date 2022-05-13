import os
import pandas as pd
from typing import List

import config as c
from models.frame import Frame


class DataBase:

    FRAMEIDX, TRACKIDX = "FrameIdx", "TrackIdx"
    CAM_LEFT, CAM_RIGHT = "Cam_Left", "Cam_Right"
    X_LEFT, X_RIGHT, Y = "X_Left", "X_Right", "Y"

    @staticmethod
    def build_tracks_database(data: List[Frame]) -> pd.DataFrame:
        # TODO: make this more efficient!
        if len(data) == 0:
            # return empty DataFrame with correct columns & index
            df = pd.DataFrame(columns=[DataBase.X_LEFT, DataBase.X_RIGHT, DataBase.Y,
                                       DataBase.FRAMEIDX, DataBase.TRACKIDX])
            df.set_index([DataBase.FRAMEIDX, DataBase.TRACKIDX], inplace=True)
            return df

        mini_dfs = {}
        for fr in data:
            fr_idx = fr.id
            df = pd.DataFrame({track_id: (kp_l.pt[0], kp_r.pt[0], kp_l.pt[1]) for (kp_l, kp_r), track_id
                               in fr.match_to_track_id.items()}).T  # a DataFrame of shape Nx3
            df.rename(columns={0: DataBase.X_LEFT, 1: DataBase.X_RIGHT, 2: DataBase.Y}, inplace=True)
            mini_dfs[fr_idx] = df
        db = pd.concat(mini_dfs)
        db.index.set_names([DataBase.FRAMEIDX, DataBase.TRACKIDX], inplace=True)
        return db

    @staticmethod
    def build_cameras_database(data: List[Frame]) -> pd.DataFrame:
        cameras = {}
        for fr in data:
            cameras[fr.id] = {DataBase.CAM_LEFT: fr.left_camera, DataBase.CAM_RIGHT: fr.right_camera}
        camera_df = pd.DataFrame.from_dict(cameras, orient='index')
        camera_df.index.name = DataBase.FRAMEIDX
        return camera_df

    @staticmethod
    def from_pickle(p: str):
        """
        Reads a pandas DataFrame/Series object from pkl file.
        :param p: str -> the pickle filename or full path. If only a filename is provided, defaults to the DATA_WRITE
            directory (taken from `config` file)
        :return: pd.DataFrame or pd.Series
        """
        # TODO: assert correct structure before returning
        if not p.endswith(".pkl"):
            p += ".pkl"
        if os.path.dirname(p) != "":
            return pd.read_pickle(p)
        fullpath = os.path.join(c.DATA_WRITE_PATH, p)
        return pd.read_pickle(fullpath)

