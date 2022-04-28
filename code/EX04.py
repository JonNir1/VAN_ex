import os
import cv2
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import utils as u
from models.database import DataBase
from service.trajectory_processor import estimate_trajectory
from logic.db_adapter import DBAdapter

###################################
#        PRELIMINARY CHECK        #
#  make sure the new impl. works  #
###################################

real_traj = u.read_trajectory()
all_frames, est_traj, elapsed = estimate_trajectory(verbose=False)

fig, ax = plt.subplots()
fig.suptitle('KITTI Trajectories')
n = est_traj.shape[1]
markers_sizes = np.ones((n,))
markers_sizes[[i for i in range(n) if i % 50 == 0]] = 15
markers_sizes[0], markers_sizes[-1] = 50, 50
ax.scatter(est_traj[0], est_traj[2], marker="o", s=markers_sizes, c=est_traj[1], cmap="gray", label="estimated")
ax.scatter(real_traj[0], real_traj[2], marker="x", s=markers_sizes, c=real_traj[1], label="ground truth")
ax.set_title("Trajectories")
ax.legend(loc='best')
fig.set_figwidth(10)
plt.show()

##################################
#          Question 4.1          #
#          build the DB          #
##################################

dba = DBAdapter(all_frames)
dba.to_pickle()  # save data to file

##################################
#         Question 4.2           #
#         present stats          #
##################################


def calculate_trackings_statistics(dbadapter: DBAdapter):
    n_frames = dbadapter.db.index.get_level_values(DataBase.FRAMEIDX).unique().size
    n_tracks = dbadapter.db.index.get_level_values(DataBase.TRACKIDX).unique().size
    trk_lengths = dbadapter.get_track_lengths()
    trks_per_fr = dbadapter.db.groupby(level=DataBase.FRAMEIDX).size()
    return n_frames, n_tracks, trk_lengths, trks_per_fr


num_frames, num_tracks, track_lengths, tracks_per_frame = calculate_trackings_statistics(dba)
print(f"Tracking Statistics:\n\tNum Frames: {num_frames}\n\tNum Tracks: {num_tracks}")
print(f"\tTrack Lengths:\tmean={track_lengths.mean():.2f}\tmin={track_lengths.min():.2f}\tmax={track_lengths.max():.2f}")
print(f"\tFrame Density:\tmean={tracks_per_frame.mean():.2f}\tmin={tracks_per_frame.min():.2f}\tmax={tracks_per_frame.max():.2f}")


##################################
#         Question 4.3           #
#         display track          #
##################################


def display_track(dbadapter: DBAdapter, min_len: int = 10):
    # TODO
    pass


#######################################
#           Question 4.4              #
#         Connectivity Graph          #
#######################################



######################################
#           Question 4.5             #
#          % Inliers Graph           #
######################################



###########################################
#              Question 4.6               #
#         Track Length Histogram          #
###########################################




