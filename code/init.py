import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

import config as c
import utils as u
from models.directions import Side
from models.camera import Camera
from service.frame_processor import FrameProcessor
from logic.db_adapter import DBAdapter
from models.database import DataBase
from service.TrajectoryOptimizer2 import TrajectoryOptimizer2
from logic.PoseGraph import PoseGraph

# change matplotlib's backend
matplotlib.use("webagg")

#####################################

start = time.time()

# need to read camera matrices manually, which is very ugly but necessary because of WSL :(
first_cam_path = "/mnt/c/Users/nirjo/Documents/University/Masters/Computer Vision Aided Navigation/VAN_ex/dataset/sequences/00/calib.txt"
K, _, Mright = u.read_first_camera_matrices(first_cam_path)
Camera._K = K
Camera._RightRotation = Mright[:, :3]
Camera._RightTranslation = Mright[:, 3:]

########################################
#        Initial Estimation            #
#  1. Calculate PnP absolute Cameras   #
#  2. Drop Short Tracks                #
#  3. Convert to Relative Cameras      #
########################################

fp = FrameProcessor(verbose=True)
all_frames, elapsed_part1 = fp.process_frames()
dba = DBAdapter(all_frames)

long_tracks_db = dba.prune_short_tracks(4)  # only use tracks of length 4 or longer

relative_cams = []
for i, cam in enumerate(dba.cameras_db["Cam_Left"]):
    if i == 0:
        relative_cams.append(cam)
        continue
    back_cam = dba.cameras_db["Cam_Left"][i-1]
    back_R, back_t = back_cam.get_rotation_matrix(), back_cam.get_translation_vector()
    front_R, front_t = cam.get_rotation_matrix(), cam.get_translation_vector()
    R = front_R @ back_R.T
    t = front_t - R @ back_t
    ext = np.hstack([R, t])
    rel_cam = Camera(i, Side.LEFT, ext)
    relative_cams.append(rel_cam)

left_cams = pd.Series(relative_cams, name="Cam_Left")
right_cams = left_cams.apply(lambda cam: cam.calculate_right_camera())
right_cams.name = "Cam_Right"
relative_cameras_db = pd.DataFrame([left_cams, right_cams]).T
relative_cameras_db.index.name = DataBase.FRAMEIDX

dba.tracks_db = long_tracks_db
dba.cameras_db = relative_cameras_db
dba.to_pickle()


#####################################
#        Bundle Adjustment          #
#####################################

traj_opt = TrajectoryOptimizer2(tracks=dba.tracks_db, relative_cams=dba.cameras_db[DataBase.CAM_LEFT])
traj_opt.optimize(verbose=True)
traj_cameras = pd.Series(traj_opt.extract_all_relative_cameras())
traj_cameras.to_pickle(os.path.join(c.DATA_WRITE_PATH, f"bundle_cameras_{datetime.now().strftime('%d%m%Y_%H%M')}.pkl"))


################################
#        Loop Closure          #
################################

pg = PoseGraph(traj_opt.bundles)
loop_stats = pg.optimize_with_loops(verbose=True)
pg_cameras = pg.extract_relative_cameras()
loop_stats.to_pickle(os.path.join(c.DATA_WRITE_PATH, f"loop_stats_{datetime.now().strftime('%d%m%Y_%H%M')}.pkl"))
pg_cameras.to_pickle(os.path.join(c.DATA_WRITE_PATH, f"posegraph_cameras_{datetime.now().strftime('%d%m%Y_%H%M')}.pkl"))


elapsed = time.time() - start
print(f"\nTotal Runtime:\t{elapsed:.2f} s")
