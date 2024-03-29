import os
import time
import random
import cv2
import numpy as np
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


# detect and identify matches and split to inliers/outliers based on y-axis distance
def extract_keypoints_and_inliers(img0, img1):
    kps0, desc0 = DEFAULT_DETECTOR.detectAndCompute(img0, None)
    kps1, desc1 = DEFAULT_DETECTOR.detectAndCompute(img1, None)
    matches = DEFAULT_MATCHER.match(desc0, desc1)

    # Stereo rectified images should have the same Y coordinate for matched keypoints
    inlier_matches, outlier_matches = [], []
    for m in matches:
        kp0, kp1 = kps0[m.queryIdx], kps1[m.trainIdx]
        y_distance = abs(kp0.pt[1] - kp1.pt[1])
        if y_distance <= MaxYDistanceForStereoInliers:
            inlier_matches.append(m)
        else:
            outlier_matches.append(m)

    # sorting matches by the first kp idx (useful later for consensus matches)
    inlier_matches = sorted(inlier_matches, key=lambda match: match.queryIdx)
    outlier_matches = sorted(outlier_matches, key=lambda match: match.queryIdx)
    return kps0, desc0, kps1, desc1, inlier_matches, outlier_matches


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
preprocess_pair_0_0 = extract_keypoints_and_inliers(img0_left, img0_right)
keypoints0_left, descriptors0_left, keypoints0_right, descriptors0_right, inliers_0_0, _ = preprocess_pair_0_0
point_cloud_0 = cv_triangulate_matched_points(keypoints0_left, keypoints0_right, inliers_0_0,
                                              K, R0_left, t0_left, R0_right, t0_right)

# triangulate keypoints from stereo pair 1:
preprocess_pair_1_1 = extract_keypoints_and_inliers(img1_left, img1_right)
keypoints1_left, descriptors1_left, keypoints1_right, descriptors1_right, inliers_1_1, _ = preprocess_pair_1_1

# triangulate pair_1 inliers based on projection matrices from pair_0
# why? because they asked us to in class.
point_cloud_1_with_camera_0 = cv_triangulate_matched_points(keypoints1_left, keypoints1_right, inliers_1_1,
                                                            K, R0_left, t0_left, R0_right, t0_right)

# QUESTION 2
# find matches in the first tracking pair (img0, img1)
# sorting to make consensus-match faster
tracking_matches = sorted(DEFAULT_MATCHER.match(descriptors0_left, descriptors1_left), key=lambda m: m.queryIdx)


# QUESTION 3
def find_consensus_matches_indices(back_inliers, front_inliers, tracking_inliers):
    # TODO: make this more efficient (from o(n^2) to O(n*logn)), see:
    #  https://stackoverflow.com/questions/71764536/most-efficient-way-to-match-2d-coordinates-in-python
    #
    # Returns a list of 3-tuples indices, representing the idx of the consensus match
    # in each of the 3 original match-lists
    consensus = []
    back_inliers_left_idx = [m.queryIdx for m in back_inliers]
    front_inliers_left_idx = [m.queryIdx for m in front_inliers]
    for idx in range(len(tracking_inliers)):
        back_left_kp_idx = tracking_inliers[idx].queryIdx
        front_left_kp_idx = tracking_inliers[idx].trainIdx
        try:
            idx_of_back_left_kp_idx = back_inliers_left_idx.index(back_left_kp_idx)
            idx_of_front_left_kp_idx = front_inliers_left_idx.index(front_left_kp_idx)
        except ValueError:
            continue
        consensus.append(tuple([idx_of_back_left_kp_idx, idx_of_front_left_kp_idx, idx]))
    return consensus


consensus_match_indices_0_1 = find_consensus_matches_indices(inliers_0_0, inliers_1_1, tracking_matches)


def calculate_front_camera_matrix(cons_matches, back_points_cloud,
                                  front_inliers, front_kps_left, intrinsic_matrix):
    # Use cv2.solvePnP to compute the front-left camera's extrinsic matrix
    # based on at least 4 consensus matches and their corresponding 2D & 3D positions
    num_samples = len(cons_matches)
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
        cons_match = cons_matches[i]
        points_3D[i] = back_points_cloud[cons_match[0]]
        front_left_matched_kp_idx = front_inliers[cons_match[1]].queryIdx
        points_2D[i] = front_kps_left[front_left_matched_kp_idx].pt

    success, rotation, translation = cv2.solvePnP(objectPoints=points_3D,
                                                  imagePoints=points_2D,
                                                  cameraMatrix=intrinsic_matrix,
                                                  distCoeffs=None,
                                                  flags=cv2.SOLVEPNP_EPNP)
    return success, cv2.Rodrigues(rotation)[0], translation


is_success, R1_left, t1_left = calculate_front_camera_matrix(random.sample(consensus_match_indices_0_1, 4),
                                                             point_cloud_0, inliers_1_1, keypoints1_left, K)
print(f"The extrinsic Camera Matrix for left1:\nR = {R1_left}\nt = {t1_left}")


def calculate_right_camera_matrix(R_left, t_left, right_R0, right_t0):
    assert right_R0.shape == (3, 3) and R_left.shape == (3, 3)
    assert right_t0.shape == (3, 1) or right_t0.shape == (3,)
    assert t_left.shape == (3, 1) or t_left.shape == (3,)
    right_t0 = right_t0.reshape((3, 1))
    t_left = t_left.reshape((3, 1))

    front_right_Rot = right_R0 @ R_left
    front_right_trans = right_R0 @ t_left + right_t0
    assert front_right_Rot.shape == (3, 3) and front_right_trans.shape == (3, 1)
    return front_right_Rot, front_right_trans


def calculate_camera_locations(back_left_R, back_left_t, right_R0, right_t0,
                               cons_matches, back_points_cloud, front_inliers, front_kps_left, intrinsic_matrix):
    # Returns a 4x3 np array representing the 3D position of the 4 cameras,
    # in coordinates of the back_left camera (hence the first line should be np.zeros(3))
    back_right_R, back_right_t = calculate_right_camera_matrix(back_left_R, back_left_t, right_R0, right_t0)
    is_success = False
    while not is_success:
        cons_sample = random.sample(cons_matches, 4)
        is_success, front_left_R, front_left_t = calculate_front_camera_matrix(cons_sample, back_points_cloud,
                                                                               front_inliers, front_kps_left,
                                                                               intrinsic_matrix)
    front_right_R, front_right_t = calculate_right_camera_matrix(front_left_R, front_left_t, back_right_R, back_right_t)

    back_right_coordinates = - back_right_R.T @ back_right_t
    front_left_coordinates = - front_left_R.T @ front_left_t
    front_right_coordinates = - front_right_R.T @ front_right_t
    return np.array([np.zeros((3, 1)), back_right_coordinates,
                     front_left_coordinates, front_right_coordinates]).reshape((4, 3))


camera_coordinates = calculate_camera_locations(R0_left, t0_left, R0_right, t0_right,consensus_match_indices_0_1,
                                                point_cloud_0, inliers_1_1, keypoints1_left, K)
print(f"The camera locations in left_0 coordinates:\n{camera_coordinates}")

# plot the positions of the 4 cameras
plt.clf(), plt.cla()
fig = plt.figure()
ax = fig.add_subplot()
colors = ['b', 'g', 'r', 'c']
for j in range(len(colors)):
    position = 'L' if j % 2 == 0 else 'R'
    ax.scatter(camera_coordinates[j][0], camera_coordinates[j][2],
               c=colors[3], s=20, marker='o', label=f'{position}{j//2}')
ax.set_xlim(-0.2, 0.8)
ax.set_ylim(-0.1, 1.2)
ax.set_xlabel('X')
ax.set_ylabel('Z')
plt.legend()
plt.title("Camera Positions\n(y=0)")
plt.show()


# QUESTION 4
def calculate_pixels_for_3d_points(points_cloud_3d, intrinsic_matrix, Rs, ts):
    """
    Takes a collection of 3D points in the world and calculates their projection on the cameras' planes.
    The 3D points should be an array of shape 3xN.
    $Rs and $ts are rotation matrices and translation vectors and should both have length M.

    return: a Mx2xN np array of (p_x, p_y) pixel coordinates for each camera
    """
    assert len(Rs) == len(ts),\
        "Number of rotation matrices and translation vectors must be equal"
    assert points_cloud_3d.shape[0] == 3 or points_cloud_3d.shape[1] == 3, \
        f"Must provide a 3D points matrix, input has shape {points_cloud_3d.shape}"
    if points_cloud_3d.shape[0] != 3:
        points_cloud_3d = points_cloud_3d.T

    num_cameras = len(Rs)
    num_points = points_cloud_3d.shape[1]
    pixels = np.zeros((num_cameras, 2, num_points))
    for i in range(num_cameras):
        R, t = Rs[i], ts[i]
        t = np.reshape(t, (3, 1))
        projections = intrinsic_matrix @ (R @ points_cloud_3d + t)  # non normalized homogeneous coordinates of shape 3xN
        hom_coordinates = projections / (projections[2] + Epsilon)  # add epsilon to avoid 0 division
        pixels[i] = hom_coordinates[:2]
    return pixels


def extract_actual_consensus_pixels(cons_matches, back_inliers, front_inliers,
                                    back_left_kps, back_right_kps, front_left_kps, front_right_kps):
    # Returns a 4x2xN array containing the 2D pixels of all consensus-matched keypoints
    back_left_pixels, back_right_pixels = [], []
    front_left_pixels, front_right_pixels = [], []
    for m in cons_matches:
        # cons_matches is a list of tuples of indices: (back_inliers_idx, front_inlier_idx, tracking_match_idx)
        single_back_inlier, single_front_inlier = back_inliers[m[0]], front_inliers[m[1]]

        back_left_point = back_left_kps[single_back_inlier.queryIdx].pt
        back_left_pixels.append(np.array(back_left_point))

        back_right_point = back_right_kps[single_back_inlier.trainIdx].pt
        back_right_pixels.append(np.array(back_right_point))

        front_left_point = front_left_kps[single_front_inlier.queryIdx].pt
        front_left_pixels.append(np.array(front_left_point))

        front_right_point = front_right_kps[single_front_inlier.trainIdx].pt
        front_right_pixels.append(np.array(front_right_point))

    back_left_pixels = np.array(back_left_pixels).T
    back_right_pixels = np.array(back_right_pixels).T
    front_left_pixels = np.array(front_left_pixels).T
    front_right_pixels = np.array(front_right_pixels).T
    return np.array([back_left_pixels, back_right_pixels, front_left_pixels, front_right_pixels])


def find_supporter_indices_for_model(cons_3d_points, actual_pixels, intrinsic_matrix, Rs, ts, max_distance: int = 2):
    """
    Find supporters for the model ($Rs & $ts) our of all consensus-matches.
    A supporter is a consensus match that has a calculated projection (based on $Rs & $ts) that is "close enough"
    to it's actual keypoints' pixels in all four images. The value of "close enough" is the argument $max_distance

    Returns a list of consensus matches that support the current model.
    """

    # make sure we have a Nx3 cloud:
    cloud_shape = cons_3d_points.shape
    assert cloud_shape[0] == 3 or cloud_shape[1] == 3, "Argument $cons_3d_points is not a 3D-points array"
    if cloud_shape[1] != 3:
        cons_3d_points = cons_3d_points.T

    # calculate pixels for all four cameras and make sure it has correct shape
    calculated_pixels = calculate_pixels_for_3d_points(cons_3d_points.T, intrinsic_matrix, Rs, ts)
    assert actual_pixels.shape == calculated_pixels.shape

    # find indices that are no more than $max_distance apart on all 4 projections
    euclidean_distances = np.linalg.norm(actual_pixels - calculated_pixels, ord=2, axis=1)
    supporting_indices = np.where((euclidean_distances <= max_distance).all(axis=0))[0]
    return supporting_indices


real_pixels = extract_actual_consensus_pixels(consensus_match_indices_0_1, inliers_0_0, inliers_1_1,
                                              keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right)
consensus_3d_points = point_cloud_0[[m[0] for m in consensus_match_indices_0_1]]
supporter_indices = find_supporter_indices_for_model(consensus_3d_points, real_pixels, K, Rs, ts)
supporters = [consensus_match_indices_0_1[idx] for idx in supporter_indices]

# plot the supporters on img0_left and img1_left:
supporting_tracking_matches = [tracking_matches[idx] for (_, _, idx) in supporters]
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


# QUESTION 5:
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


def build_model(consensus_match_idxs, points_cloud_3d, front_inliers, kps_front_left,
                intrinsic_matrix, back_left_rot, back_left_trans, R0_right, t0_right, use_random=True):
    # calculate the model (R & t of each camera) based on
    # the back-left camera and the [R|t] transformation to Right camera
    back_right_rot, back_right_trans = calculate_right_camera_matrix(back_left_rot, back_left_trans, R0_right, t0_right)
    is_success = False
    while not is_success:
        sample_consensus_matches = random.sample(consensus_match_idxs, 4) if use_random else consensus_match_idxs
        is_success, front_left_rot, front_left_trans = calculate_front_camera_matrix(sample_consensus_matches,
                                                                                     points_cloud_3d, front_inliers,
                                                                                     kps_front_left, intrinsic_matrix)
    front_right_rot, front_right_trans = calculate_right_camera_matrix(front_left_rot, front_left_trans,
                                                                       R0_right, t0_right)
    Rs = [back_left_rot, back_right_rot, front_left_rot, front_right_rot]
    ts = [back_left_trans, back_right_trans, front_left_trans, front_right_trans]
    return Rs, ts


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
    success_prob = 0.99
    outlier_prob = 0.99  # this value is updated while running RANSAC
    num_iterations = calculate_number_of_iteration_for_ransac(0.99, outlier_prob, 4)

    prev_supporters_indices = []
    cons_3d_points = points_cloud_3d[[m[0] for m in cons_match_idxs]]
    actual_pixels = extract_actual_consensus_pixels(cons_match_idxs, back_inliers, front_inliers,
                                                    kps_back_left, kps_back_right, kps_front_left, kps_front_right)
    if verbose:
        print(f"Starting RANSAC with {num_iterations} iterations.")
    while num_iterations > 0:
        Rs, ts = build_model(cons_match_idxs, points_cloud_3d, front_inliers, kps_front_left,
                             intrinsic_matrix, back_left_rot, back_left_trans, R0_right, t0_right, use_random=True)
        supporters_indices = find_supporter_indices_for_model(cons_3d_points, actual_pixels,
                                                              intrinsic_matrix, Rs, ts)

        if len(supporters_indices) > len(prev_supporters_indices):
            prev_supporters_indices = supporters_indices
            outlier_prob = 1 - len(prev_supporters_indices) / len(cons_match_idxs)
            num_iterations = calculate_number_of_iteration_for_ransac(0.99, outlier_prob, 4)
            if verbose:
                print(f"\tRemaining iterations: {num_iterations}\n\t\t" +
                      f"Number of Supporters: {len(prev_supporters_indices)}")
        else:
            num_iterations -= 1
            if verbose and num_iterations % 100 == 0:
                print(f"Remaining iterations: {num_iterations}\n\t\t" +
                      f"Number of Supporters: {len(prev_supporters_indices)}")

    # at this point we have a good model (Rs & ts) and we can refine it based on all supporters
    if verbose:
        print("Refining RANSAC results...")
    while True:
        curr_supporters = [cons_match_idxs[idx] for idx in prev_supporters_indices]
        Rs, ts = build_model(curr_supporters, points_cloud_3d, front_inliers, kps_front_left,
                             intrinsic_matrix, Rs[0], ts[0], R0_right, t0_right, use_random=False)
        supporters_indices = find_supporter_indices_for_model(cons_3d_points, actual_pixels, intrinsic_matrix, Rs, ts)
        if len(supporters_indices) > len(prev_supporters_indices):
            # we can refine the model even further
            prev_supporters_indices = supporters_indices
        else:
            # no more refinement, exit the loop
            break

    # finished, we can return the model
    curr_supporters = [cons_match_idxs[idx] for idx in prev_supporters_indices]
    elapsed = time.time() - start_time
    if verbose:
        print(f"RANSAC finished in {elapsed:.2f} seconds\n\tNumber of Supporters: {len(curr_supporters)}")
    return Rs, ts, curr_supporters



mR, mt, sup = estimate_projection_matrices_with_ransac(point_cloud_0, consensus_match_indices_0_1,
                                                       inliers_0_0, inliers_1_1,
                                                       keypoints0_left, keypoints0_right,
                                                       keypoints1_left, keypoints1_right,
                                                       K, R0_left, t0_left, R0_right, t0_right,
                                                       verbose=True)


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


# create scatter plot of the two point clouds:
point_cloud_0_transformed_to_1 = transform_coordinates(point_cloud_0.T, mR[2], mt[2])
plt.clf(), plt.cla()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(point_cloud_0.T[0], point_cloud_0.T[2],
             point_cloud_0.T[1], c='b', s=2.5, marker='o', label='left0')
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
supporting_tracking_matches = [tracking_matches[idx] for (_a, _b, idx) in consensus_match_indices_0_1
                               if (_a, _b, idx) in sup]
non_supporting_tracking_matches = [tracking_matches[idx] for (_a, _b, idx) in consensus_match_indices_0_1
                                   if (_a, _b, idx) not in sup]
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

#################


# Question 6:
NUM_FRAMES = 3450  # total number of stereo-images in our KITTI dataset


def estimate_complete_trajectory(num_frames: int = NUM_FRAMES, verbose=False):
    start_time, minutes_counter = time.time(), 0
    if verbose:
        print(f"Starting to process trajectory for {num_frames} tracking-pairs...")

    # load initiial cameras:
    K, M1, M2 = read_cameras()
    R0_left, t0_left = M1[:, :3], M1[:, 3:]
    R0_right, t0_right = M2[:, :3], M2[:, 3:]
    Rs_left, ts_left = [R0_left], [t0_left]

    # load first pair:
    img0_l, img0_r = read_images(0)
    back_pair_preprocess = extract_keypoints_and_inliers(img0_l, img0_r)
    back_left_kps, back_left_desc, back_right_kps, back_right_desc, back_inliers, _ = back_pair_preprocess

    for idx in range(1, num_frames):
        back_left_R, back_left_t = Rs_left[-1], ts_left[-1]
        back_right_R, back_right_t = calculate_right_camera_matrix(back_left_R, back_left_t, R0_right, t0_right)
        points_cloud_3d = cv_triangulate_matched_points(back_left_kps, back_right_kps, back_inliers,
                                                        K, back_left_R, back_left_t, back_right_R, back_right_t)

        # run the estimation on the current pair:
        front_left_img, front_right_img = read_images(idx)
        front_pair_preprocess = extract_keypoints_and_inliers(front_left_img, front_right_img)
        front_left_kps, front_left_desc, front_right_kps, front_right_desc, front_inliers, _ = front_pair_preprocess
        track_matches = sorted(DEFAULT_MATCHER.match(back_left_desc, front_left_desc),
                               key=lambda match: match.queryIdx)
        consensus_indices = find_consensus_matches_indices(back_inliers, front_inliers, track_matches)
        curr_Rs, curr_ts, _ = estimate_projection_matrices_with_ransac(points_cloud_3d, consensus_indices, back_inliers,
                                                                       front_inliers, back_left_kps, back_right_kps,
                                                                       front_left_kps, front_right_kps, K,
                                                                       back_left_R, back_left_t, R0_right, t0_right,
                                                                       verbose=False)
        # print update if needed:
        curr_minute = int((time.time() - start_time) / 60)
        if verbose and curr_minute > minutes_counter:
            minutes_counter = curr_minute
            print(f"\tProcessed {idx} tracking-pairs in {minutes_counter} minutes")

        # update variables for the next pair:
        Rs_left.append(curr_Rs[2])
        ts_left.append(curr_ts[2])
        back_left_kps, back_left_desc = front_left_kps, front_left_desc
        back_right_kps, back_right_desc = front_right_kps, front_right_desc
        back_inliers = front_inliers

    total_elapsed = time.time() - start_time
    if verbose:
        total_minutes = total_elapsed / 60
        print(f"Finished running for all tracking-pairs. Total runtime: {total_minutes:.2f} minutes")
    return Rs_left, ts_left, total_elapsed


def read_poses():
    Rs, ts = [], []
    file_path = os.path.join(os.getcwd(), r'dataset\poses\00.txt')
    f = open(file_path, 'r')
    for i, line in enumerate(f.readlines()):
        mat = np.array(line.split(), dtype=float).reshape((3, 4))
        Rs.append(mat[:, :3])
        ts.append(mat[:, 3:])
    return Rs, ts


def calculate_trajectory(Rs, ts):
    assert len(Rs) == len(ts),\
        "number of rotation matrices and translation vectors mismatch"
    num_samples = len(Rs)
    trajectory = np.zeros((num_samples, 3))
    for i in range(num_samples):
        R, t = Rs[i], ts[i]
        trajectory[i] -= (R.T @ t).reshape((3,))
    return trajectory


def compute_trajectory_and_distance(num_frames: int = NUM_FRAMES, verbose: bool = False):
    if verbose:
        print(f"\nCALCULATING TRAJECTORY FOR {num_frames} IMAGES\n")
    all_R, all_t, elapsed = estimate_complete_trajectory(num_frames, verbose=verbose)
    estimated_trajectory = calculate_trajectory(all_R, all_t)
    poses_R, poses_t = read_poses()
    ground_truth_trajectory = calculate_trajectory(poses_R[:num_frames], poses_t[:num_frames])
    distances = np.linalg.norm(estimated_trajectory - ground_truth_trajectory, ord=2, axis=1)
    return estimated_trajectory, ground_truth_trajectory, distances


estimated_trajectory, ground_truth_trajectory, distances = compute_trajectory_and_distance(num_frames=NUM_FRAMES,
                                                                                           verbose=True)

fig, axes = plt.subplots(1, 2)
fig.suptitle('KITTI Trajectories')
n = estimated_trajectory.T[0].shape[0]
markers_sizes = np.ones((n,))
markers_sizes[[i for i in range(n) if i % 50 == 0]] = 15
markers_sizes[0], markers_sizes[-1] = 50, 50
axes[0].scatter(estimated_trajectory.T[0], estimated_trajectory.T[2],
                   marker="o", s=markers_sizes, c=estimated_trajectory.T[1], cmap="gray", label="estimated")
axes[0].scatter(ground_truth_trajectory.T[0], ground_truth_trajectory.T[2],
                   marker="x", s=markers_sizes, c=ground_truth_trajectory.T[1], label="ground truth")
axes[0].set_title("Trajectories")
axes[0].legend(loc='best')

axes[1].scatter([i for i in range(n)], distances, c='k', marker='*', s=1)
axes[1].set_title("Euclidean Distance between Trajectories")
fig.set_figwidth(10)
plt.show()



########################################
#       TESTING - NOT HW RELATED       #
########################################

plt.clf()
num_images = [500, 1000, 1500, 2000, 3000, 3450]
est_trajects, gt_trajects, dists = [], [], []
fig, axes = plt.subplots(len(num_images), 2)
fig.suptitle('KITTI Trajectories for Varying Data-Set Size')
for i in range(len(num_images)):
    n = num_images[i]
    est_traj, gt_traj, curr_dists = compute_trajectory_and_distance(n, True)
    est_trajects.append(est_traj)
    gt_trajects.append(gt_traj)
    dists.append(curr_dists)

    markers_sizes = np.ones((n,))
    markers_sizes[[i for i in range(n) if i % 50 == 0]] = 15
    markers_sizes[0], markers_sizes[-1] = 50, 50

    axes[i][0].scatter(est_traj.T[0], est_traj.T[2],
                       marker="o", s=markers_sizes, c=est_traj.T[1], cmap="gray", label="estimated")
    axes[i][0].scatter(gt_traj.T[0], gt_traj.T[2],
                       marker="x", s=markers_sizes, c=gt_traj.T[1], label="ground truth")
    title_left = f"Trajectories for\n{n} Images" if i == 0 else f"{n} Images"
    axes[i][0].set_title(title_left)

    axes[i][1].scatter([i for i in range(n)], curr_dists, c='k', marker='*', s=1)
    title_right = "Distance between Trajectories" if i == 0 else ""
    axes[i][1].set_title(title_right)

    if i == len(num_images) - 1:
        axes[i][0].legend(loc='best')

fig.tight_layout()
fig.set_figheight(18)
plt.show()




