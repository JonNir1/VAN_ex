import pandas as pd

from models.frame import Frame


class DataBase:

    FRAME, FRAMEIDX, TRACKIDX = "Frame", "FrameIdx", "TrackIdx"
    CAM_LEFT, CAM_RIGHT = "Cam_Left", "Cam_Right"
    X_LEFT, X_RIGHT, Y = "X_Left", "X_Right", "Y"

    @staticmethod
    def build_database(data: list[Frame]) -> pd.DataFrame:
        # TODO: make this more efficient!
        if len(data) == 0:
            # return empty DataFrame with correct columns & index
            df = pd.DataFrame(columns=[DataBase.X_LEFT, DataBase.X_RIGHT, DataBase.Y, DataBase.FRAME,
                                       DataBase.FRAMEIDX, DataBase.TRACKIDX])
            df.set_index([DataBase.FRAMEIDX, DataBase.TRACKIDX], inplace=True)
            return df

        mini_dfs = {}
        for fr in data:
            fr_idx = fr.get_id()
            df = pd.DataFrame({tr_id: (kp_l.pt[0], kp_r.pt[0], kp_l.pt[1])
                               for tr_id, (kp_l, kp_r) in fr.get_tracks().items()}).T  # a DataFrame of shape Nx3
            df.rename(columns={0: DataBase.X_LEFT, 1: DataBase.X_RIGHT, 2: DataBase.Y}, inplace=True)
            mini_dfs[fr_idx] = df
        db = pd.concat(mini_dfs)
        db.index.set_names([DataBase.FRAMEIDX, DataBase.TRACKIDX], inplace=True)
        db[DataBase.FRAME] = [f for f in data for j in range(len(f.get_tracks()))]  # store Frame objects in DB for serialization
        return db



