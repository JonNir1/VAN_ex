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
all_frames, est_traj, _ = estimate_trajectory(verbose=False)
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

long_track, length = dba.sample_track_idx_with_length(10, 14)
track_data = dba.db.xs(long_track, level=DataBase.TRACKIDX)
frame_ids = track_data.index.get_level_values(DataBase.FRAMEIDX).tolist()

fig, axes = plt.subplots(len(frame_ids), 2)
fig.suptitle(f"Frames {min(frame_ids)}-{max(frame_ids)}\nTrack{long_track}")
for i, frame_idx in enumerate(frame_ids):
    img_l, img_r = u.read_image_pair(frame_idx)
    x_l, x_r, y, _ = track_data.xs(frame_idx)
    axes[i][0].imshow(img_l, cmap='gray', vmin=0, vmax=255)
    axes[i][0].scatter(x_l, y, s=4, c='orange')
    axes[i][0].axis('off')
    axes[i][1].imshow(img_r, cmap='gray', vmin=0, vmax=255)
    axes[i][1].scatter(x_r, y, s=4, c='orange')
    axes[i][1].axis('off')
plt.subplots_adjust(wspace=-0.65, hspace=0.3, top=0.9, bottom=0.02)
plt.show()


#######################################
#           Question 4.4              #
#         Connectivity Graph          #
#######################################


def calculate_connectivity(dbadapter: DBAdapter, shift: int = 1) -> pd.Series:
    # For all Frames, returns the amount of shared tracks between any Frame i and frame i+shift
    frame_indices = dbadapter.db.index.get_level_values(DataBase.FRAMEIDX)
    last_frame_to_check = frame_indices.max() - shift + 1
    shared_tracks_count = dict()
    for fr_idx in range(last_frame_to_check):
        try:
            shared_tracks_count[fr_idx] = dbadapter.get_shared_tracks(fr_idx, fr_idx + shift).count()
        except KeyError:
            break
    return pd.Series(shared_tracks_count, name=f"Shared_Tracks_(i+{shift})")


one_connectivity = calculate_connectivity(dba, shift=1)
ax = one_connectivity.plot.line(color='b')
ax.set_title("Connectivity")
ax.set_xlabel("FrameIdx")
ax.set_ylabel("Outgoing Tracks")
plt.show()
print(f"Mean Connectivity: {one_connectivity.mean() :.2f}")

######################################
#           Question 4.5             #
#          % Inliers Graph           #
######################################
# TODO


###########################################
#              Question 4.6               #
#         Track Length Histogram          #
###########################################

ax = track_lengths.plot.hist()
ax.set_title("Track Length Histogram")
ax.set_xlabel("Track Length")
ax.set_ylabel("Count")
plt.show()

#########################################
#            Question 4.7               #
#          Reprojection Error           #
#########################################

from models.directions import Side
from models.camera import Camera
from logic.triangulation import triangulate

# TODO: this should be it's own class

cam0_l, cam0_r = Camera.read_first_cameras()
K = cam0_l.intrinsic_matrix
Rs, ts = u.read_poses()

long_track, length = dba.sample_track_idx_with_length(10, 14)
track_data = dba.db.xs(long_track, level=DataBase.TRACKIDX)
frame_ids = track_data.index.get_level_values(DataBase.FRAMEIDX).tolist()
last_frame_id = max(frame_ids)
_, _, _, frame = track_data.xs(last_frame_id)
frame_match = frame.get_match(track_id=long_track)

last_cam_left = Camera(idx=last_frame_id, side=Side.LEFT, intrinsic_mat=K,
                       extrinsic_mat=Camera.calculate_extrinsic_matrix(Rs[last_frame_id], ts[last_frame_id]))
last_cam_right = last_cam_left.calculate_right_camera()
point_3d = triangulate(matches=[frame_match], left_cam=last_cam_left, right_cam=last_cam_right)

# TODO - project on back images and calculate errs



