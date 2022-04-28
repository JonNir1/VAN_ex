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

start = time.time()
real_traj = u.read_trajectory()
# all_frames, est_traj, _ = estimate_trajectory(verbose=False)

real_traj = real_traj[:, :50]
all_frames, est_traj, _ = estimate_trajectory(num_frames=50, verbose=True)

error = np.linalg.norm(est_traj - real_traj, ord=2, axis=0)
elapsed = time.time() - start
print(f"FINISHED PROCESSING TRAJECTORY IN {(elapsed / 60):.2f} MINUTES")

fig, axes = plt.subplots(1, 2)
fig.suptitle('KITTI Trajectories')
n = est_traj.shape[1]
markers_sizes = np.ones((n,))
markers_sizes[[i for i in range(n) if i % 50 == 0]] = 15
markers_sizes[0], markers_sizes[-1] = 50, 50
axes[0].scatter(est_traj[0], est_traj[2], marker="o", s=markers_sizes, c=est_traj[1], cmap="gray", label="estimated")
axes[0].scatter(real_traj[0], real_traj[2], marker="x", s=markers_sizes, c=real_traj[1], label="ground truth")
axes[0].set_title("Trajectories")
axes[0].legend(loc='best')

axes[1].scatter([i for i in range(n)], error, c='k', marker='*', s=1)
axes[1].set_title("Euclidean Distance between Trajectories")
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


def calculate_tracking_statistics(dbadapter: DBAdapter):
    n_frames = dbadapter.db.index.get_level_values(DataBase.FRAMEIDX).unique().size
    n_tracks = dbadapter.db.index.get_level_values(DataBase.TRACKIDX).unique().size
    trk_lengths = dbadapter.get_track_lengths()
    trks_per_fr = dbadapter.db.groupby(level=DataBase.FRAMEIDX).size()
    return n_frames, n_tracks, trk_lengths, trks_per_fr


num_frames, num_tracks, track_lengths, tracks_per_frame = calculate_tracking_statistics(dba)
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


def calculate_connectivity_graph(dbadapter: DBAdapter, shift: int = 1) -> pd.Series:
    # For all Frames, returns the amount of shared tracks between any Frame i and frame i+shift
    frame_indices = dbadapter.db.index.get_level_values(DataBase.FRAMEIDX)
    last_frame_to_check = frame_indices.max() - shift

    shared_tracks_count = dict()
    for fr_idx in range(last_frame_to_check):
        try:
            shared_tracks_count[fr_idx] = dbadapter.get_shared_tracks(fr_idx, fr_idx + shift).count()
        except KeyError:
            break
    return pd.Series(shared_tracks_count, name=f"Shared_Tracks_(i+{shift})")


one_connectivity = calculate_connectivity_graph(dba, shift=1)
ax = one_connectivity.plot()
plt.show()


######################################
#           Question 4.5             #
#          % Inliers Graph           #
######################################



###########################################
#              Question 4.6               #
#         Track Length Histogram          #
###########################################

ax = track_lengths.plot(kind='hist')
plt.show()


