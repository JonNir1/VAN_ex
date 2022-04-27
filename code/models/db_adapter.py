import os
import pandas as pd
import pickle as pkl
from datetime import datetime

import config as c
from models.frame import Frame


class DBAdapter:

    FRAMEIDX = "FrameIdx"
    TRACKIDX = "TrackIdx"
    CAM_LEFT, CAM_RIGHT = "Cam_Left", "Cam_Right"
    X_LEFT, X_RIGHT, Y = "X_Left", "X_Right", "Y"

    @staticmethod
    def build_database(data: list[Frame]) -> pd.DataFrame:
        # TODO: make this more efficient!
        mini_dfs = {}
        for fr in data:
            fr_idx = fr.get_id()
            df = pd.DataFrame({tr.get_id() : (kp_l.pt[0], kp_r.pt[0], kp_l.pt[1])
                               for tr, (kp_l, kp_r) in fr.get_tracks().items()}).T  # a DataFrame of shape Nx3
            df.rename(columns={0: DBAdapter.X_LEFT, 1: DBAdapter.X_RIGHT, 2: DBAdapter.Y}, inplace=True)
            mini_dfs[fr_idx] = df
        db = pd.concat(mini_dfs)
        db.index.set_names([DBAdapter.FRAMEIDX, DBAdapter.TRACKIDX], inplace=True)
        db[DBAdapter.CAM_LEFT] = [f.left_camera.extrinsic_matrix for f in data for j in range(len(f.get_tracks()))]
        db[DBAdapter.CAM_RIGHT] = [f.right_camera.extrinsic_matrix for f in data for j in range(len(f.get_tracks()))]
        return db

    @staticmethod
    def to_pickle(db: pd.DataFrame, filename: str = None):
        if not os.path.isdir(c.DATA_WRITE_PATH):
            os.makedirs(c.DATA_WRITE_PATH)
        filename = filename if filename is not None else f"db_{datetime.now().strftime('%d%m%Y_%H%M')}"
        fullpath = os.path.join(c.DATA_WRITE_PATH, filename + ".pkl")
        with open(fullpath, 'wb') as f:
            pkl.dump(db, f, protocol=-1)

    @staticmethod
    def read_pickle(filename: str) -> pd.DataFrame:
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        fullpath = os.path.join(c.DATA_WRITE_PATH, filename)
        return pd.read_pickle(fullpath)



