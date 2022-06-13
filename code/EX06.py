import os
import gtsam
import numpy as np
import pandas as pd
from gtsam.utils import plot
import matplotlib
from matplotlib import pyplot as plt

import config as c
import utils as u
from models.camera import Camera
from models.database import DataBase
from logic.trajectory import read_ground_truth_trajectory
from logic.db_adapter import DBAdapter
from logic.PoseGraph import PoseGraph
from service.TrajectoryOptimizer2 import TrajectoryOptimizer2

# change matplotlib's backend
matplotlib.use("webagg")

##################################
#          Load Data             #
##################################

K, M_left, M_right = u.read_first_camera_matrices()
Camera._K = K
Camera._RightRotation = M_right[:, :3]
Camera._RightTranslation = M_right[:, 3:]

traj_gt = read_ground_truth_trajectory()

dba = DBAdapter(data=[])
dba.tracks_db = pd.read_pickle(os.path.join(c.DATA_WRITE_PATH, 'long_tracks.pkl'))
dba.cameras_db = pd.read_pickle(os.path.join(c.DATA_WRITE_PATH, 'relative_cameras.pkl'))

# perform bundle adjustment from EX05
traj_opt = TrajectoryOptimizer2(tracks=dba.tracks_db, relative_cams=dba.cameras_db[DataBase.CAM_LEFT])
traj_opt.optimize(verbose=True)

#################################
#        Question 6.1           #
#      extract covariance       #
#################################

bundle0 = traj_opt.bundles[0]
marginals0 = gtsam.Marginals(bundle0.graph, bundle0.optimized_estimates)

start_frame_idx = min(bundle0.frame_symbols.keys())
start_frame_symbol = bundle0.frame_symbols[start_frame_idx]
end_frame_idx = max(bundle0.frame_symbols.keys())
end_frame_symbol = bundle0.frame_symbols[end_frame_idx]
end_frame_pose = bundle0.optimized_estimates.atPose3(end_frame_symbol)

keys = gtsam.KeyVector()
keys.append(start_frame_symbol)
keys.append(end_frame_symbol)
marginal_cov = marginals0.jointMarginalCovariance(keys).fullMatrix()
info_mat0 = np.linalg.inv(marginal_cov)
relative_cov0 = np.linalg.inv(info_mat0[-6:, -6:])

print(end_frame_pose)
print(relative_cov0)

plt.clf()
plot.plot_trajectory(fignum=0, values=bundle0.optimized_estimates, scale=1,
                     title="First Bundle Trajectory", marginals=marginals0)
plt.show()


#################################
#         Question 6.2          #
#       Build Pose Graph        #
#################################

pg = PoseGraph(traj_opt.bundles)
pre_err = pg.error
print(f"Pre-Optimization Error:\t{pre_err:.5f}")
pg.optimize()
post_err = pg.error
print(f"Post-Optimization Error:\t{post_err:.5f}")

# plot keyframe locations provided as initial estimates
plt.clf()
plot.plot_trajectory(fignum=1, values=pg._initial_estimates, scale=1, title="Pose Graph :: Initial Estimates")
plt.show()

# plot keyframe locations after optimization
plt.clf()
plot.plot_trajectory(fignum=2, values=pg._optimized_estimates, scale=1, title="Pose Graph :: Optimization Results")
plt.show()

# plot keyframe locations after optimization - with covariances
marginals = gtsam.Marginals(pg._graph, pg._optimized_estimates)
plt.clf()
plot.plot_trajectory(fignum=3, values=pg._optimized_estimates, scale=1, marginals=marginals,
                     title="Pose Graph :: Optimization Results with Covariances")
plt.show()
