# Adapted from https://github.com/borglab/gtsam/blob/develop/python/gtsam/tests/test_StereoVOExample.py
# if everything works as expected, the script should print the follwing two lines:
# "Poses Equal: True"
# "Point Equal: True"

import numpy as np
import gtsam

# create keys for variables
x1 = gtsam.symbol('x', 1)
x2 = gtsam.symbol('x', 2)
l1 = gtsam.symbol('l', 1)
l2 = gtsam.symbol('l', 2)
l3 = gtsam.symbol('l', 3)

# Create graph container & add constraint on starting-pose
graph = gtsam.NonlinearFactorGraph()
first_pose = gtsam.Pose3()
graph.add(gtsam.NonlinearEqualityPose3(x1, first_pose))

# create noise model
K = gtsam.Cal3_S2Stereo(1000, 1000, 0, 320, 240, 0.2)
noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))

# add pose 1
graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(520, 480, 440), noise_model, x1, l1, K))
graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(120, 80, 440), noise_model, x1, l2, K))
graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(320, 280, 140), noise_model, x1, l3, K))

# add pose 2
graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(570, 520, 490), noise_model, x2, l1, K))
graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(70, 20, 490), noise_model, x2, l2, K))
graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(320, 270, 115), noise_model, x2, l3, K))

# create initial estimate
initial_estimate = gtsam.Values()
initial_estimate.insert(x1, first_pose)

initial_estimate.insert(x2, gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.1, -0.1, 1.1)))
expected_l1 = gtsam.Point3(1, 1, 5)
initial_estimate.insert(l1, expected_l1)
initial_estimate.insert(l2, gtsam.Point3(-1, 1, 5))
initial_estimate.insert(l3, gtsam.Point3(0, -0.5, 5))

# optimize
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
result = optimizer.optimize()

# check equality for first pose and landmark
pose_x1 = result.atPose3(x1)
print(f"Poses Equal: {pose_x1.equals(first_pose, 1e-4)}")

point_l1 = result.atPoint3(l1)
print(f"Point Equal: {np.allclose(point_l1, expected_l1, 1e-4)}")
