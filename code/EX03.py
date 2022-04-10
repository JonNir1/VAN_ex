import os
import random
import time
import cv2
import numpy as np
from typing import Callable, Any
from matplotlib import pyplot as plt

DATA_PATH = os.path.join(os.getcwd(), r'dataset\sequences\00')
DEFAULT_DETECTOR = cv2. SIFT_create()
# DEFAULT_MATCHER = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
DEFAULT_MATCHER = cv2.FlannBasedMatcher(indexParams=dict(algorithm=0, trees=5),
                                        searchParams=dict(checks=50))

MaxYDistanceForStereoInliers = 2
Epsilon = 1e-10


def read_images(idx: int):
    image_name = "{:06d}.png".format(idx)
    img0 = cv2.imread(DATA_PATH + '\\image_0\\' + image_name, 0)
    img1 = cv2.imread(DATA_PATH + '\\image_1\\' + image_name, 0)
    return img0, img1


def read_cameras():
    with open(DATA_PATH + '\\calib.txt') as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


img0_left, img0_right = read_images(0)
img1_left, img1_right = read_images(1)
K, Ext0_left, Ext0_right = read_cameras()  # intrinsic & extrinsic camera Matrices
R0_left, t0_left = Ext0_left[:, :3], Ext0_left[:, 3:]
R0_right, t0_right = Ext0_right[:, :3], Ext0_right[:, 3:]


def extract_keypoints_and_inliers(img0, img1, inlier_identifier: Callable[[Any, Any, Any], bool]):
    kps0, desc0 = DEFAULT_DETECTOR.detectAndCompute(img0, None)
    kps1, desc1 = DEFAULT_DETECTOR.detectAndCompute(img1, None)
    matches = DEFAULT_MATCHER.match(desc0, desc1)
    inlier_matches, outlier_matches = [], []
    for m in matches:
        if inlier_identifier(m, kps0, kps1):
            inlier_matches.append(m)
        else:
            outlier_matches.append(m)

    # sorting matches by the first kp index (useful later for consensus matches)
    inlier_matches = sorted(inlier_matches, key=lambda match: match.queryIdx)
    outlier_matches = sorted(outlier_matches, key=lambda match: match.queryIdx)
    return kps0, desc0, kps1, desc1, inlier_matches, outlier_matches


# Stereo rectified images should have the same Y coordinate for matched keypoints
def _stereo_inlier_identifier(match, kps0, kps1) -> bool:
    kp0, kp1 = kps0[match.queryIdx], kps1[match.trainIdx]
    y_distance = abs(kp0.pt[1] - kp1.pt[1])
    return y_distance <= MaxYDistanceForStereoInliers


# tracking images accept all matches as inliers
def _tracking_inlier_identifier(match, kps0, kps1) -> bool:
    return True


# cv2 triangulation
def cv_triangulate_matched_points(kps0, kps1, inlier_matches,
                                  K, R_back_left, t_back_left, R_back_right, t_back_right):
    num_matches = len(inlier_matches)
    # extract the (x,y) coordinates of each match into two 2xN np.arrays
    matched_kps0_coordinates = np.array([kps0[inlier_matches[i].queryIdx].pt for i in range(num_matches)])
    matched_kps1_coordinates = np.array([kps1[inlier_matches[i].trainIdx].pt for i in range(num_matches)])

    # calculate the camera's projection matrix (a 4x3 matrix)
    proj_mat_left = K @ np.hstack((R_back_left, t_back_left))
    proj_mat_right = K @ np.hstack((R_back_right, t_back_right))

    # use cv2 triangulation function
    X_4d = cv2.triangulatePoints(proj_mat_left, proj_mat_right, matched_kps0_coordinates.T, matched_kps1_coordinates.T)
    X_4d /= (X_4d[3] + 1e-10)  # homogenize; add small epsilon to prevent division by 0

    # return only the 3d coordinates
    return X_4d[:-1].T



# QUESTION 1
# triangulate keypoints from stereo pair 0:
preprocess_pair_0_0 = extract_keypoints_and_inliers(img0_left, img0_right, _stereo_inlier_identifier)
keypoints0_left, descriptors0_left, keypoints0_right, descriptors0_right, inliers_0_0, outliers_0_0 = preprocess_pair_0_0
point_cloud_0 = cv_triangulate_matched_points(keypoints0_left, keypoints0_right, inliers_0_0,
                                              K, R0_left, t0_left, R0_right, t0_right)

# triangulate keypoints from stereo pair 1:
preprocess_pair_1_1 = extract_keypoints_and_inliers(img1_left, img1_right, _stereo_inlier_identifier)
keypoints1_left, descriptors1_left, keypoints1_right, descriptors1_right, inliers_1_1, outliers_1_1 = preprocess_pair_1_1

# triangulate pair_1 inliers based on projection matrices from pair_0
# why? because they asked us to in class.
point_cloud_1_with_camera_0 = cv_triangulate_matched_points(keypoints1_left, keypoints1_right, inliers_1_1,
                                                            K, R0_left, t0_left, R0_right, t0_right)

# QUESTION 2
# find matches in the first tracking pair (img0, img1)
tracking_matches = DEFAULT_MATCHER.match(descriptors0_left, descriptors1_left)
tracking_matches = sorted(tracking_matches, key=lambda match: match.queryIdx)  # sorting to make consensus-match faster


# QUESTION 3
def find_consensus_matches_indices(back_inliers, front_inliers, tracking_inliers):
    # efficient (O(n*logn)) way to find consensus matches between 3 matching-lists
    # matching-lists should be sorted according to the first kp's index (match.queryIdx)
    # see https://stackoverflow.com/questions/71764536/most-efficient-way-to-match-2d-coordinates-in-python
    #
    # Returns a list of 3-tuples indices, representing the index of the consensus match
    # in each of the 3 original match-lists
    consensus = []
    back_inliers_left_idx = [m.queryIdx for m in back_inliers]
    front_inliers_left_idx = [m.queryIdx for m in front_inliers]
    for idx in range(len(tracking_inliers)):
        back_left_kp_idx = tracking_matches[idx].queryIdx
        front_left_kp_idx = tracking_matches[idx].trainIdx
        try:
            idx_of_back_left_kp_idx = back_inliers_left_idx.index(back_left_kp_idx)
            idx_of_front_left_kp_idx = front_inliers_left_idx.index(front_left_kp_idx)
        except ValueError:
            continue
        consensus.append(tuple([idx_of_back_left_kp_idx, idx_of_front_left_kp_idx, idx]))
    return consensus


consensus_match_indices = find_consensus_matches_indices(inliers_0_0, inliers_1_1, tracking_matches)


def _calculate_front_left_extrinsic_camera_matrix(sampled_cons_matches, back_points_cloud,
                                                  front_inliers, front_kps_left,
                                                  intrinsic_camera_matrix):
    # Use cv2.solvePnP to compute the front-left camera's extrinsic matrix
    # based on at least 4 consensus matches and their corresponding 2D & 3D positions
    num_samples = len(sampled_cons_matches)
    if num_samples < 4:
        raise ValueError(f"Must provide at least 4 sampled consensus-matches, {num_samples} given")
    cloud_shape = back_points_cloud.shape
    assert cloud_shape[0] == 3 or cloud_shape[1] == 3, "Argument $back_points_cloud is not a 3D array"
    if cloud_shape[1] != 3:
        back_points_cloud = back_points_cloud.T  # making sure we have shape Nx3 for solvePnP
    points_3D = np.zeros((num_samples, 3))
    points_2D = np.zeros((num_samples, 2))

    # populate the arrays
    for i in range(num_samples):
        cons_match = sampled_cons_matches[i]
        points_3D[i] = back_points_cloud[cons_match[0]]
        front_left_matched_kp_idx = front_inliers[cons_match[1]].queryIdx
        points_2D[i] = front_kps_left[front_left_matched_kp_idx].pt

    success, rotation, translation = cv2.solvePnP(objectPoints=points_3D,
                                                  imagePoints=points_2D,
                                                  cameraMatrix=intrinsic_camera_matrix,
                                                  distCoeffs=None,
                                                  flags=cv2.SOLVEPNP_EPNP)
    if not success:
        return np.zeros((3, 3)), np.zeros((3, 1))
    return cv2.Rodrigues(rotation)[0], translation


def calculate_front_left_extrinsic_camera_matrix_with_random(cons_matches, back_points_cloud,
                                                             front_inliers, front_kps_left,
                                                             intrinsic_camera_matrix, n: int = 4):
    random_consensus_matches = random.sample(cons_matches, n)
    return _calculate_front_left_extrinsic_camera_matrix(random_consensus_matches, back_points_cloud,
                                                         front_inliers, front_kps_left,
                                                         intrinsic_camera_matrix)



R1_left, t1_left = calculate_front_left_extrinsic_camera_matrix_with_random(consensus_match_indices, point_cloud_0,
                                                                            inliers_1_1, keypoints1_left, K)
print(f"The extrinsic Camera Matrix for left1:\nR = {R1_left}\nt = {t1_left}")


def _calculate_right_extrinsic_camera_matrix(R_right_0, t_right_0,
                                             R_left, t_left):
    assert R_right_0.shape == (3, 3) and R_left.shape == (3, 3)
    assert t_right_0.shape == (3, 1) or t_right_0.shape == (3,)
    assert t_left.shape == (3, 1) or t_left.shape == (3,)
    t_right_0 = t_right_0.reshape((3, 1))
    t_left = t_left.reshape((3, 1))

    front_right_Rot = R_right_0 @ R_left
    front_right_trans = R_right_0 @ t_left + t_right_0
    assert front_right_Rot.shape == (3, 3) and front_right_trans.shape == (3, 1)
    return front_right_Rot, front_right_trans


def calculate_camera_locations(back_right_R, back_right_t,
                               front_left_R, front_left_t):
    # Returns a 4x3 np array representing the 3D position of the 4 cameras,
    # in coordinates of the back_left camera (hence the first line should be np.zeros(3))
    back_right_coordinates = - back_right_R.T @ back_right_t
    front_left_coordinates = - front_left_R.T @ front_left_t

    front_right_R, front_right_t = _calculate_right_extrinsic_camera_matrix(back_right_R, back_right_t, front_left_R,
                                                                            front_left_t)
    front_right_coordinates = - front_right_R.T @ front_right_t
    return np.array([np.zeros((3, 1)), back_right_coordinates,
                     front_left_coordinates, front_right_coordinates]).reshape((4, 3))


R1_right, t1_right = _calculate_right_extrinsic_camera_matrix(R0_right, t0_right, R1_left, t1_left)
camera_coordinates = calculate_camera_locations(R0_right, t0_right, R1_left, t1_left)
print(f"The camera locations in left_0 coordinates:\n{camera_coordinates}")

# plot the positions of the 4 cameras
plt.clf(), plt.cla()
fig = plt.figure()
ax = fig.add_subplot()
colors = ['b', 'g', 'r', 'c']
for i in range(len(colors)):
    position = 'L' if i % 2 == 0 else 'R'
    ax.scatter(camera_coordinates[i][0], camera_coordinates[i][2],
               c=colors[i], s=20, marker='o', label=f'{position}{i}')
ax.set_xlim(-0.1, 0.8)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel('X')
ax.set_ylabel('Z')
plt.legend()
plt.title("Camera Positions\n(y=0)")
plt.show()


# QUESTION 4
def calculate_pixels_for_3d_points(points_cloud_3d, K, R, t):
    """
    Takes a collection of 3D points in the world and calculates their projection on the camera's plane
    param points_cloud_3d: a 3xN np.array of world coordinates
    param K: intrinsic camera matrix
    param R: extrinsic rotation matrix
    param t: extrinsic translation vector

    return: a 2xN array of (p_x, p_y) pixel coordinates
    """
    assert points_cloud_3d.shape[0] == 3,\
        "3D points matrix should have the (x,y,z) coordinates as rows (i.e. shape 3xN)"
    projections = K @ (R @ points_cloud_3d + t)  # non normalized homogeneous coordinates of shape 3xN
    hom_coordinates = projections / (projections[2] + Epsilon)  # add epsilon to avoid 0 division
    # assert np.isclose(hom_coordinates[2], np.ones(hom_coordinates[2].shape)).all()
    return hom_coordinates[:2]


def calculate_pixels_for_3d_points_multiple_cameras(points_cloud_3d, K, Rs, ts):
    """
    Takes a collection of 3D points in the world and calculates their projection on the cameras' planes.
    The 3D points should be an array of shape 3xN.
    $Rs and $ts are rotation matrices and translation vectors and should both have length M.

    return: a Mx2xN np array of (p_x, p_y) pixel coordinates for each camera
    """
    assert len(Rs) == len(ts), "Number of rotation matrices and translation vectors must be equal"
    num_cameras = len(Rs)
    num_points = points_cloud_3d.shape[1]
    pixels = np.zeros((num_cameras, 2, num_points))
    for i in range(num_cameras):
        R, t = Rs[i], ts[i]
        pixels[i] = calculate_pixels_for_3d_points(points_cloud_3d, K, R, t)
    return pixels


def extract_single_consensus_pixels(single_cons_match, back_inliers, front_inliers,
                                    kps_back_l, kps_back_r, kps_front_l, kps_front_r):
    # Returns a 4x2 array of pixels of matched keypoints on each camera
    single_back_inlier = back_inliers[single_cons_match[0]]
    single_front_inlier = front_inliers[single_cons_match[1]]
    back_left_pixels = kps_back_l[single_back_inlier.queryIdx].pt
    back_right_pixels = kps_back_r[single_back_inlier.trainIdx].pt
    front_left_pixels = kps_front_l[single_front_inlier.queryIdx].pt
    front_right_pixels = kps_front_r[single_front_inlier.trainIdx].pt
    return np.array([back_left_pixels, back_right_pixels, front_left_pixels, front_right_pixels])


def extract_all_consensus_pixels(cons_matches, back_inliers, front_inliers,
                                 kps_back_l, kps_back_r, kps_front_l, kps_front_r):
    # Returns a 4x2xN array containing the 2D pixels of all consensus-matched keypoints
    back_left_pixels, back_right_pixels = [], []
    front_left_pixels, front_right_pixels = [], []
    for m in cons_matches:
        pixels = extract_single_consensus_pixels(m, back_inliers, front_inliers,
                                                 kps_back_l, kps_back_r, kps_front_l, kps_front_r)
        back_left_pixels.append(pixels[0])
        back_right_pixels.append(pixels[1])
        front_left_pixels.append(pixels[2])
        front_right_pixels.append(pixels[3])

    # arrange return value as a 4x2xN array
    back_left_pixels = np.array(back_left_pixels).T
    back_right_pixels = np.array(back_right_pixels).T
    front_left_pixels = np.array(front_left_pixels).T
    front_right_pixels = np.array(front_right_pixels).T
    return np.array([back_left_pixels, back_right_pixels, front_left_pixels, front_right_pixels])


def find_consensus_supporters(points_cloud_3d, cons_matches, actual_pixels,
                              K, Rs, ts, max_distance: int = 2):
    # make sure we have a Nx3 cloud:
    cloud_shape = points_cloud_3d.shape
    assert cloud_shape[0] == 3 or cloud_shape[1] == 3, "Argument $points_cloud_3d is not a 3D array"
    if cloud_shape[1] != 3:
        points_cloud_3d = points_cloud_3d.T

    # extract only the 3D points that have a match between all 4 images
    points_cloud_with_consensus = points_cloud_3d[[m[0] for m in cons_matches]]
    calculated_pixels = calculate_pixels_for_3d_points_multiple_cameras(points_cloud_with_consensus.T, K, Rs, ts)
    assert actual_pixels.shape == calculated_pixels.shape

    # find indices that are no more than $max_distance apart on all 4 projections
    euclidean_distances = np.linalg.norm(actual_pixels - calculated_pixels, ord=2, axis=1)
    supporting_indices = np.where((euclidean_distances <= max_distance).all(axis=0))[0]
    supporting_matches = [cons_matches[idx] for idx in supporting_indices]
    return supporting_matches


real_pixels = extract_all_consensus_pixels(consensus_match_indices, inliers_0_0, inliers_1_1,
                                           keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right)
supporting_consensus_indices = find_consensus_supporters(point_cloud_0, consensus_match_indices, real_pixels,
                                                         K, [R0_left, R0_right, R1_left, R1_right],
                                                         [t0_left, t0_right, t1_left, t1_right], max_distance=2)

supporting_tracking_matches = [tracking_matches[idx] for (_, _, idx) in supporting_consensus_indices]
non_supporting_tracking_matches = [m for m in tracking_matches if m not in supporting_tracking_matches]

supporting_pixels_back = [keypoints0_left[i].pt for i in [m.queryIdx for m in supporting_tracking_matches]]
supporting_pixels_front = [keypoints1_left[i].pt for i in [m.trainIdx for m in supporting_tracking_matches]]
non_supporting_pixels_back = [keypoints0_left[i].pt for i in [m.queryIdx for m in non_supporting_tracking_matches]]
non_supporting_pixels_front = [keypoints1_left[i].pt for i in [m.trainIdx for m in non_supporting_tracking_matches]]

plt.clf(), plt.cla()
fig, axes = plt.subplots(2)

axes[1].scatter([x for (x, y) in supporting_pixels_back],
                [y for (x, y) in supporting_pixels_back],
                s=4, c='orange', marker='*', label='supporter')
axes[1].scatter([x for (x, y) in non_supporting_pixels_back],
                [y for (x, y) in non_supporting_pixels_back],
                s=1, c='c', marker='o', label='non-sup.')
axes[1].imshow(img0_left, cmap='gray', vmin=0, vmax=255)
axes[1].axis('off')

axes[0].scatter([x for (x, y) in supporting_pixels_front],
                [y for (x, y) in supporting_pixels_front],
                s=4, c='orange', marker='*')
axes[0].scatter([x for (x, y) in non_supporting_pixels_front],
                [y for (x, y) in non_supporting_pixels_front],
                s=1, c='c', marker='o')
axes[0].imshow(img1_left, cmap='gray', vmin=0, vmax=255)
axes[0].axis('off')
plt.legend()
fig.suptitle("Supporting & Non-Supporting Matches")
plt.show()


# Question 5:
def calculate_number_of_iteration_for_ransac(p: float, e: float, s: int) -> int:
    """
    Calculate how many iterations of RANSAC are required to get good enough results,
    i.e. for a set of size $s, with outlier probability $e and success probability $p
    we need N > log(1-$p) / log(1-(1-$e)^$s)

    :param p: float -> required success probability (0 < $p < 1)
    :param e: float -> probability to be outlier (0 < $e < 1)
    :param s: int -> minimal set size (s > 0)
    :return: N: float -> number of iterations
    """
    assert s > 0, "minimal set size must be a positive integer"
    nom = np.log(1 - p)
    denom = np.log(1 - np.power(1-e, s))
    return int(nom / denom) + 1


def build_and_estimate_model(consensus_match_idxs, points_cloud_3d, actual_pixels,
                             front_inliers, kps_front_left, intrinsic_matrix,
                             back_left_rot, back_left_trans, R0_right, t0_right, use_random=True):
    if use_random:
        front_left_rot, front_left_trans = calculate_front_left_extrinsic_camera_matrix_with_random(consensus_match_idxs,
                                                                                                    points_cloud_3d,
                                                                                                    front_inliers,
                                                                                                    kps_front_left,
                                                                                                    intrinsic_matrix,
                                                                                                    n=4)
    else:
        front_left_rot, front_left_trans = _calculate_front_left_extrinsic_camera_matrix(consensus_match_idxs,
                                                                                         points_cloud_3d,
                                                                                         front_inliers, kps_front_left,
                                                                                         intrinsic_matrix)

    back_right_rot, back_right_trans = _calculate_right_extrinsic_camera_matrix(R0_right, t0_right, back_left_rot,
                                                                                back_left_trans)
    front_right_rot, front_right_trans = _calculate_right_extrinsic_camera_matrix(R0_right, t0_right, front_left_rot,
                                                                                  front_left_trans)
    Rs = [back_left_rot, back_right_rot, front_left_rot, front_right_rot]
    ts = [back_left_trans, back_right_trans, front_left_trans, front_right_trans]
    supporters = find_consensus_supporters(points_cloud_3d, consensus_match_idxs, actual_pixels,
                                           intrinsic_matrix, Rs, ts, max_distance=2)
    return Rs, ts, supporters


def estimate_projection_matrices_with_ransac(points_cloud_3d, cons_match_idxs,
                                             back_inliers, front_inliers,
                                             kps_back_left, kps_back_right,
                                             kps_front_left, kps_front_right,
                                             intrinsic_matrix,
                                             back_left_rot, back_left_trans,
                                             R0_right, t0_right,
                                             verbose: bool = False):
    """
    Implement RANSAC algorithm to estimate extrinsic matrix of the two front cameras,
    based on the two back cameras, the consensus-matches and the 3D points-cloud of the back pair.

    Returns the best fitting model:
        - Rs - rotation matrices of 4 cameras
        - ts - translation vectors of 4 cameras
        - supporters - subset of consensus-matches that support this model,
            i.e. projected keypoints are no more than 2 pixels away from the actual keypoint
    """
    start_time = time.time()
    success_prob, minimal_set_size = 0.99, 4
    outlier_prob = 0.99  # this value is updated while running RANSAC
    num_iterations = calculate_number_of_iteration_for_ransac(success_prob, outlier_prob, 4)

    best_supporters = []
    actual_pixels = extract_all_consensus_pixels(cons_match_idxs, back_inliers, front_inliers,
                                                 kps_back_left, kps_back_right, kps_front_left, kps_front_right)
    if verbose:
        print(f"Starting RANSAC with {num_iterations} iterations.")
    while num_iterations > 0:
        Rs, ts, supporters = build_and_estimate_model(cons_match_idxs, points_cloud_3d, actual_pixels, front_inliers,
                                                      kps_front_left, intrinsic_matrix, back_left_rot, back_left_trans,
                                                      R0_right, t0_right, use_random=True)
        if len(supporters) > len(best_supporters):
            best_supporters = supporters
            outlier_prob = 1 - len(best_supporters) / len(cons_match_idxs)
            num_iterations = calculate_number_of_iteration_for_ransac(success_prob, outlier_prob, minimal_set_size)
            if verbose:
                print(f"\tRemaining iterations: {num_iterations}\n\t\tNumber of Supporters: {len(best_supporters)}")
        else:
            num_iterations -= 1
            if verbose and num_iterations % 100 == 0:
                print(f"Remaining iterations: {num_iterations}\n\t\t" +
                      f"Number of Supporters: {len(best_supporters)}")

    # at this point we have a good model (Rs & ts) and we can refine it based on all supporters
    if verbose:
        print("Refining RANSAC results...")
    actual_pixels = extract_all_consensus_pixels(best_supporters, back_inliers, front_inliers,
                                                 kps_back_left, kps_back_right, kps_front_left, kps_front_right)
    while True:
        Rs, ts, supporters = build_and_estimate_model(best_supporters, points_cloud_3d, actual_pixels, front_inliers,
                                                      kps_front_left, intrinsic_matrix, back_left_rot, back_left_trans,
                                                      R0_right, t0_right, use_random=False)
        if len(best_supporters) < len(supporters):
            # we can refine the model even further
            best_supporters = supporters
        else:
            # no more refinement, exit the loop
            break

    # finished, we can return the model
    elapsed = time.time() - start_time
    if verbose:
        print(f"RANSAC finished in {elapsed:.2f} seconds\n\tNumber of Supporters: {len(best_supporters)}")
    return Rs, ts, best_supporters


model_R, model_t, model_supporters = estimate_projection_matrices_with_ransac(point_cloud_0, consensus_match_indices,
                                                                              inliers_0_0, inliers_1_1, keypoints0_left,
                                                                              keypoints0_right, keypoints1_left,
                                                                              keypoints1_right, K, R0_left, t0_left,
                                                                              R0_right, t0_right, verbose=True)


def transform_coordinates(points_3d, R, t):
    input_shape = points_3d.shape
    assert input_shape[0] == 3 or input_shape[1] == 3, \
        f"can only operate on matrices of shape 3xN or Nx3, provided {input_shape}"
    if input_shape[0] != 3:
        points_3d = points_3d.T  # making sure we are working with a 3xN array

    assert t.shape == (3, 1) or t.shape == (3,), \
        f"translation vector must be of size 3, provided {t.shape}"
    if t.shape != (3, 1):
        t = np.reshape(t, (3, 1))  # making sure we are using a 3x1 vector
    assert R.shape == (3, 3), f"rotation matrix must be of shape 3x3, provided {R.shape}"
    transformed = R @ points_3d + t
    assert transformed.shape == points_3d.shape
    return transformed


point_cloud_0 = point_cloud_0.T
point_cloud_0_transformed_to_1 = transform_coordinates(point_cloud_0, model_R[2], model_t[2])

# create scatter plot of the two point clouds:
plt.clf(), plt.cla()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(point_cloud_0[0], point_cloud_0[2],
             point_cloud_0[1], c='b', s=2.5, marker='o', label='left0')
ax.scatter3D(point_cloud_0_transformed_to_1[0], point_cloud_0_transformed_to_1[2],
             point_cloud_0_transformed_to_1[1], c='r', s=2.5, marker='o', label='left1')
ax.set_xlim(-10, 10)
ax.set_ylim(-2, 20)
ax.set_zlim(-4, 16)
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')
plt.legend()
plt.show()

# plot the supporters and non-supporters on the first tracking-pair
supporting_tracking_matches = [tracking_matches[idx] for (_a, _b, idx) in consensus_match_indices
                               if (_a, _b, idx) in model_supporters]
non_supporting_tracking_matches = [tracking_matches[idx] for (_a, _b, idx) in consensus_match_indices
                                   if (_a, _b, idx) not in model_supporters]
supporting_pixels_back = [keypoints0_left[i].pt for i in [m.queryIdx for m in supporting_tracking_matches]]
supporting_pixels_front = [keypoints1_left[i].pt for i in [m.trainIdx for m in supporting_tracking_matches]]
non_supporting_pixels_back = [keypoints0_left[i].pt for i in [m.queryIdx for m in non_supporting_tracking_matches]]
non_supporting_pixels_front = [keypoints1_left[i].pt for i in [m.trainIdx for m in non_supporting_tracking_matches]]

plt.clf(), plt.cla()
fig, axes = plt.subplots(2)

axes[1].scatter([x for (x, y) in supporting_pixels_back],
                [y for (x, y) in supporting_pixels_back],
                s=1, c='orange', marker='*', label='supporter')
axes[1].scatter([x for (x, y) in non_supporting_pixels_back],
                [y for (x, y) in non_supporting_pixels_back],
                s=1, c='c', marker='o', label='non-sup.')
axes[1].imshow(img0_left, cmap='gray', vmin=0, vmax=255)
axes[1].axis('off')

axes[0].scatter([x for (x, y) in supporting_pixels_front],
                [y for (x, y) in supporting_pixels_front],
                s=1, c='orange', marker='*')
axes[0].scatter([x for (x, y) in non_supporting_pixels_front],
                [y for (x, y) in non_supporting_pixels_front],
                s=1, c='c', marker='o')
axes[0].imshow(img1_left, cmap='gray', vmin=0, vmax=255)
axes[0].axis('off')
plt.legend()
fig.suptitle("Supporting & Non-Supporting Matches")
plt.show()

#####################################

# img0_left, img0_right = img1_left, img1_right
# img1_left, img1_right = read_images(2)
#
# keypoints0_left, descriptors0_left = keypoints1_left, descriptors1_left
# keypoints0_right, descriptors0_right = keypoints1_right, descriptors1_right
# inliers_0_0, outliers_0_0 = preprocess_pair_0_0 = inliers_1_1, outliers_1_1
# point_cloud_0 = cv_triangulate_matched_points(keypoints0_left, keypoints0_right, inliers_0_0)
#
# # triangulate keypoints from stereo pair 1:
# preprocess_pair_1_1 = extract_keypoints_and_inliers(img1_left, img1_right, _stereo_inlier_identifier)
# keypoints1_left, descriptors1_left, keypoints1_right, descriptors1_right, inliers_1_1, outliers_1_1 = preprocess_pair_1_1
#
# tracking_matches = sorted(DEFAULT_MATCHER.match(descriptors0_left, descriptors1_left),
#                           key=lambda match: match.queryIdx)
# consensus_match_indices = find_consensus_matches_indices(inliers_0_0, inliers_1_1, tracking_matches)



#####################################


# Question 6:
# NUM_FRAMES = 3450  # total number of stereo-images in our KITTI dataset

NUM_FRAMES = 3


def estimate_complete_trajectory(verbose=False):
    start_time, minutes_counter = time.time(), 0
    if verbose:
        print(f"Starting to process trajectory for {NUM_FRAMES} tracking-pairs...")

    # load initiial cameras:
    K, M1, M2 = read_cameras()
    R0_left, t0_left = M1[:, :3], M1[:, 3:]
    R0_right, t0_right = M2[:, :3], M2[:, 3:]
    Rs_left, ts_left = [R0_left], [t0_left]

    # load first pair:
    img0_l, img0_r = read_images(0)
    back_pair_preprocess = extract_keypoints_and_inliers(img0_l, img0_r, _stereo_inlier_identifier)
    back_left_kps, back_left_desc, back_right_kps, back_right_desc, back_inliers, _ = back_pair_preprocess

    print("into the loop")

    for idx in range(1, NUM_FRAMES):

        back_left_R, back_left_t = Rs_left[-1], ts_left[-1]
        back_right_R, back_right_t = _calculate_right_extrinsic_camera_matrix(R0_right, t0_right,
                                                                              back_left_R, back_left_t)
        points_cloud_3d = cv_triangulate_matched_points(back_left_kps, back_right_kps, back_inliers,
                                                        K, back_left_R, back_left_t, back_right_R, back_right_t)


        # run the estimation on the current pair:
        front_left_img, front_right_img = read_images(idx)
        front_pair_preprocess = extract_keypoints_and_inliers(front_left_img, front_right_img, _stereo_inlier_identifier)
        front_left_kps, front_left_desc, front_right_kps, front_right_desc, front_inliers, _ = front_pair_preprocess
        track_matches = sorted(DEFAULT_MATCHER.match(back_left_desc, front_left_desc),
                               key=lambda match: match.queryIdx)

        print("AAA" + str(idx))

        consensus_indices = find_consensus_matches_indices(back_inliers, front_inliers, track_matches)
        curr_Rs, curr_ts, _ = estimate_projection_matrices_with_ransac(points_cloud_3d, consensus_indices, back_inliers,
                                                                       front_inliers, back_left_kps, back_right_kps,
                                                                       front_left_kps, front_right_kps, K,
                                                                       back_left_R, back_left_t, R0_right, t0_right,
                                                                       verbose=True)


        print("BBB" + str(idx))


        front_left_R, front_left_t = curr_Rs[2], curr_ts[2]
        Rs_left.append(front_left_R)
        ts_left.append(front_left_t)

        # print update if needed:
        curr_minute = int((time.time() - start_time) / 60)
        if verbose and (curr_minute > minutes_counter or idx % 50 == 0):
            minutes_counter = curr_minute
            print(f"\tProcessed {idx} tracking-pairs in {minutes_counter} minutes")

        # update variables for the next pair:
        back_left_kps, back_left_desc = front_left_kps, front_left_desc
        back_right_kps, back_right_desc = front_right_kps, front_right_desc
        back_inliers = front_inliers

        print("CCCCC" + str(idx))

    total_elapsed = time.time() - start_time
    if verbose:
        total_minutes = int(total_elapsed / 60)
        print(f"Finished running for all tracking-pairs. Total runtime: {total_minutes:.2f} minutes")
    return Rs_left, ts_left, total_elapsed


all_R, all_t, elapsed = estimate_complete_trajectory(verbose=True)


