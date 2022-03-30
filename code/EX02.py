import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

cwd = os.getcwd()
DATA_PATH = os.path.join(cwd, r'dataset\sequences\00')
DETECTOR = cv2. SIFT_create()
MATCHER = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


def read_images(idx: int):
    image_name = "{:06d}.png".format(idx)
    img0 = cv2.imread(DATA_PATH + '\\image_0\\' + image_name, 0)
    img1 = cv2.imread(DATA_PATH + '\\image_1\\' + image_name, 0)
    return img0, img1


def extract_keypoints_and_matches(image_left, image_right):
    keypoints_left, descriptors_left = DETECTOR.detectAndCompute(image_left, None)
    keypoints_right, descriptors_right = DETECTOR.detectAndCompute(image_right, None)
    matches = MATCHER.match(descriptors_left, descriptors_right)
    return keypoints_left, keypoints_right, matches


img0_left, img0_right = read_images(0)
kp0_left, kp0_right, matches0 = extract_keypoints_and_matches(img0_left, img0_right)


# Question 1
def calc_match_distance_on_y_axis(kps_l, kps_r, matches, idx: int) -> float:
    idx_left, idx_right = matches[idx].queryIdx, matches[idx].trainIdx
    kp_left, kp_right = kps_l[idx_left], kps_r[idx_right]
    y_dist = kp_left.pt[1] - kp_right.pt[1]
    return abs(y_dist)


y_distances = [calc_match_distance_on_y_axis(kp0_left, kp0_right, matches0, i) for i in range(len(matches0))]
plt.hist(y_distances)
plt.xlabel('pixels on Y axis')
plt.ylabel('count')
plt.title('Distribution of Matches\nby Their distance along Y axis')
plt.show()

large_y_distances = [y_dist for y_dist in y_distances if y_dist > 2]
print(f"There are {(100 * len(large_y_distances) / len(y_distances)):.2f}% of matches farther than 2px apart.")


# Question 2:
def split_keypoints_by_max_y_distance(kps_l, kps_r, matches, max_distance=2):
    good_matches = []
    bad_matches = []
    for i in range(len(matches)):
        idx_left, idx_right = matches[i].queryIdx, matches[i].trainIdx
        kp_left, kp_right = kps_l[idx_left], kps_r[idx_right]
        y_dist = abs(kp_left.pt[1] - kp_right.pt[1])
        if y_dist <= max_distance:
            good_matches.append([kp_left, kp_right])
        else:
            bad_matches.append([kp_left, kp_right])
    return good_matches, bad_matches


inliers, outliers = split_keypoints_by_max_y_distance(kp0_left, kp0_right, matches0)
# the above lists give us left_kps: inliers[i][0] ; right_kps: inliers[i][1]

plt.clf(), plt.cla()
fig = plt.figure(figsize=(16, 9))
fig.add_subplot(2, 1, 1)
plt.imshow(img0_left, cmap='gray', vmin=0, vmax=255)
plt.scatter([inliers[i][0].pt[0] for i in range(len(inliers))],
            [inliers[i][0].pt[1] for i in range(len(inliers))],
            s=3, c='orange', marker='o')
plt.scatter([outliers[i][0].pt[0] for i in range(len(outliers))],
            [outliers[i][0].pt[1] for i in range(len(outliers))],
            s=3, c='c', marker='o')
plt.axis('off')

fig.add_subplot(2, 1, 2)
plt.imshow(img0_right, cmap='gray', vmin=0, vmax=255)
plt.scatter([inliers[i][1].pt[0] for i in range(len(inliers))],
            [inliers[i][1].pt[1] for i in range(len(inliers))],
            s=3, c='orange', marker='o')
plt.scatter([outliers[i][1].pt[0] for i in range(len(outliers))],
            [outliers[i][1].pt[1] for i in range(len(outliers))],
            s=3, c='c', marker='o')
plt.axis('off')
plt.show()


# Question 3:
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


K, M1, M2 = read_cameras()
M1, M2 = K@M1, K@M2  # multiply by intrinsic camera matrix


# from a given match, create the coefficients Matrix
# for the triangulation problem. The output is a 4x4 matrix.
def extract_coefficients_from_match(match_idx: int):
    kp_left, kp_right = inliers[match_idx]
    kp_left_x, kp_left_y = kp_left.pt
    kp_right_x, kp_right_y = kp_right.pt
    M1x = M1[2] * kp_left_x - M1[0]
    M1y = M1[2] * kp_left_y - M1[1]
    M2x = M2[2] * kp_right_x - M2[0]
    M2y = M2[2] * kp_right_y - M2[1]
    return np.array([M1x, M1y, M2x, M2y])


# Solve the linear least-squares problem (Mx=0) using SVD
# and take the last line of Vt that is the best solution
# then normalize the solution and extract the 3D point from homogeneous 4D vector
def triangulate_matched_points(match_idx: int):
    M = extract_coefficients_from_match(match_idx)
    U, S, Vt = np.linalg.svd(M)
    X_hom = Vt[-1] / (Vt[-1][-1] + 1e-10)  # homogenize; add small epsilon to prevent division by 0
    return X_hom[:-1]


# Use cv2 triangulation
def cv_triangulate_matched_points(inlier_matches):
    pts_left = np.array([inlier_matches[i][0].pt for i in range(len(inlier_matches))]).T
    pts_right = np.array([inlier_matches[i][1].pt for i in range(len(inlier_matches))]).T
    X_4d = cv2.triangulatePoints(M1, M2, pts_left, pts_right)
    X_4d /= (X_4d[3] + 1e-10)  # homogenize; add small epsilon to prevent division by 0
    return X_4d[:-1].T


triangulated_points = np.array([triangulate_matched_points(i)
                                for i in range(len(inliers))])
cv_triangulated_points = cv_triangulate_matched_points(inliers)
# (np.isclose(triangulated_points, cv_triangulated_points)).all()  # yields TRUE :)

# create scatter plot to compare
plt.clf(), plt.cla()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(triangulated_points.T[0], triangulated_points.T[1],
             triangulated_points.T[2], c='b', s=5, marker='o', label='LLSq')
ax.scatter3D(cv_triangulated_points.T[0], cv_triangulated_points.T[1],
             cv_triangulated_points.T[2], c='r', s=1, marker='^', label='CV2')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-5, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()
