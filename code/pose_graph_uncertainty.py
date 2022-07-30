import os
import time
import gtsam
from datetime import datetime
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib
from matplotlib import pyplot as plt

import config as c
import utils as u
from models.directions import Side
from models.camera import Camera
from service.frame_processor import FrameProcessor
from logic.db_adapter import DBAdapter
from models.database import DataBase
from models.gtsam_frame import GTSAMFrame
from service.TrajectoryOptimizer2 import TrajectoryOptimizer2
from logic.PoseGraph import PoseGraph

# change matplotlib's backend
matplotlib.use("webagg")

#####################################

# need to read camera matrices manually, which is very ugly but necessary because of WSL :(
first_cam_path = "/mnt/c/Users/nirjo/Documents/University/Masters/Computer Vision Aided Navigation/VAN_ex/dataset/sequences/00/calib.txt"
K, _, Mright = u.read_first_camera_matrices(first_cam_path)
Camera._K = K
Camera._RightRotation = Mright[:, :3]
Camera._RightTranslation = Mright[:, 3:]

dba = DBAdapter(data=[])
dba.tracks_db = pd.read_pickle(os.path.join(c.DATA_WRITE_PATH, 'tracksdb_28072022_1049.pkl'))
dba.cameras_db = pd.read_pickle(os.path.join(c.DATA_WRITE_PATH, 'camerasdb_28072022_1049.pkl'))
traj_opt = TrajectoryOptimizer2(tracks=dba.tracks_db, relative_cams=dba.cameras_db[DataBase.CAM_LEFT])
traj_opt.optimize(verbose=True)
pg = PoseGraph(traj_opt.bundles)

pre_filename = os.path.join(c.DATA_WRITE_PATH, "posegraph_adjacency_pre_optimization.pkl")
ig.Graph.write_pickle(pg._locations_graph, pre_filename)

pg.optimize_with_loops(verbose=True)
post_filename = os.path.join(c.DATA_WRITE_PATH, "posegraph_adjacency_post_optimization.pkl")
ig.Graph.write_pickle(pg._locations_graph, post_filename)

