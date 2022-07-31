import os
import time
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime

import final_project.config as c
import final_project.logic.Utils as u
import final_project.logic.GtsamUtils as gu
from final_project.models.Camera import Camera
from final_project.models.Matcher import DEFAULT_MATCHER
from final_project.models.DataBase import DataBase
from final_project.models.Trajectory import Trajectory
from final_project.logic.PoseGraph import PoseGraph
from final_project.service.InitialEstimateCalculator import IECalc
from final_project.service.BundleAdjustment import BundleAdjustment
from final_project.service.LoopClosure import close_loops

# change matplotlib's backend
matplotlib.use("webagg")

###################################################

start = time.time()


def init(N: int = c.NUM_FRAMES):
    mtchr = DEFAULT_MATCHER
    iec = IECalc(matcher=mtchr)
    frames = iec.process(num_frames=N, verbose=True)
    database = DataBase(frames).prune_short_tracks(3)
    database.to_pickle()
    return database


db = init()
db.to_pickle()
# db = DataBase.from_pickle("tracks2.pkl", "pnp_cameras2.pkl")

ba = BundleAdjustment(db._tracks_db, db._cameras_db)
ba_cameras = ba.optimize(verbose=True)
filename_suffix = f"{datetime.now().strftime('%d%m%Y_%H%M')}.pkl"
pd.Series(ba_cameras).to_pickle(os.path.join(c.DATA_WRITE_PATH, "ba_cameras" + filename_suffix))


pg = PoseGraph(ba.get_keyframe_indices(), ba_cameras, ba.extract_relative_covariances())
pg_cameras, loop_results = close_loops(pg, verbose=True)
pd.Series(pg_cameras).to_pickle(os.path.join(c.DATA_WRITE_PATH, "pg_cameras" + filename_suffix))
loop_results.to_pickle(os.path.join(c.DATA_WRITE_PATH, "loop_results" + filename_suffix))

elapsed = time.time() - start

###############

pnp_traj = Trajectory.from_relative_cameras(db._cameras_db)
ba_traj = Trajectory.from_relative_cameras(ba_cameras)
# pg_traj = Trajectory.from_relative_cameras(pg_cameras)
gt_traj = Trajectory.from_ground_truth()

pnp_dist = pnp_traj.calculate_distance(gt_traj)
ba_dist = ba_traj.calculate_distance(gt_traj)
# pg_dist = pg_traj.calculate_distance(gt_traj)  # TODO

fig, axes = plt.subplots(1, 2)
fig.suptitle('KITTI Trajectories')
axes[0].scatter(pnp_traj.X, pnp_traj.Z, marker="o", c='b', s=2, label="PnP")
axes[0].scatter(ba_traj.X, ba_traj.Z, marker="^", c='g', s=2, label="BA")
# axes[0].scatter(pg_traj.X, pg_traj.Z, marker="s", c='gold', s=4, label="PG")
axes[0].scatter(gt_traj.X, gt_traj.Z, marker="x", c='k', s=2, label="GT")
axes[0].set_title("Trajectories")
axes[0].set_xlabel("$m$")
axes[0].set_ylabel("$m$")
axes[0].legend(loc='best')

axes[1].scatter([i for i in range(c.NUM_FRAMES)], pnp_dist, c='b', marker='o', s=1, label="PnP")
axes[1].scatter([i for i in range(c.NUM_FRAMES)], ba_dist, c='g', marker='^', s=1, label="BA")
# axes[1].scatter([i for i in range(c.NUM_FRAMES)], ba_dist, c='gold', marker='s', s=1, label="PG")
axes[1].set_title("Euclidean Distance")
axes[1].set_xlabel("Frame")
axes[1].set_ylabel("$m$")
axes[1].legend(loc='best')
fig.set_figwidth(12)
plt.show()

###################

# Compare BundleAdjustment._extract_cameras1 with BundleAdjustment._extract_cameras2

opt_cams = [gu.calculate_camera_from_gtsam_pose(ba._bundles[0]._cameras.loc[0, c.OptPose])]
for i, b in enumerate(ba._bundles):
    # if i == 0:
    #     continue
    if i == 400:
        break
    R0, t0 = opt_cams[-1].R, opt_cams[-1].t
    opt_cams = opt_cams[:-1]
    start_idx = b._tracks.index.get_level_values(c.FrameIdx).min()
    start_pose = b._cameras.loc[start_idx, c.OptPose]
    for p in b._cameras[c.OptPose]:
        between_pose = start_pose.between(p)
        bundle_abs_cam = gu.calculate_camera_from_gtsam_pose(between_pose)
        new_R = bundle_abs_cam.R @ R0
        new_t = bundle_abs_cam.R @ t0 + bundle_abs_cam.t
        opt_cams.append(Camera.from_Rt(new_R, new_t))


traj = Trajectory.from_absolute_cameras(opt_cams)

ba2_dist_ba1 = traj.calculate_distance(ba_traj)
ba2_dist_gt = traj.calculate_distance(gt_traj)


plt.clf()
fig, axes = plt.subplots(1, 2)
fig.suptitle('KITTI Trajectories')
axes[0].scatter(pnp_traj.X, pnp_traj.Z, marker="o", c='b', s=2, label="PnP")
axes[0].scatter(ba_traj.X, ba_traj.Z, marker="^", c='g', s=2, label="BA")
axes[0].scatter(traj.X, traj.Z, marker="^", c='cyan', s=2, label="BA2")
axes[0].scatter(gt_traj.X, gt_traj.Z, marker="x", c='k', s=2, label="GT")
axes[0].set_title("Trajectories")
axes[0].set_xlabel("$m$")
axes[0].set_ylabel("$m$")
axes[0].legend(loc='best')

axes[1].scatter([i for i in range(c.NUM_FRAMES)], ba_dist, c='k', marker='o', s=1, label="BA1 - GT")
axes[1].scatter([i for i in range(c.NUM_FRAMES)], ba2_dist_ba1, c='b', marker='o', s=1, label="BA2 - BA1")
axes[1].scatter([i for i in range(c.NUM_FRAMES)], ba2_dist_gt, c='g', marker='^', s=1, label="BA2 - GT")
axes[1].set_title("Euclidean Distance")
axes[1].set_xlabel("Frame")
axes[1].set_ylabel("$m$")
axes[1].legend(loc='best')
fig.set_figwidth(12)
plt.show()



