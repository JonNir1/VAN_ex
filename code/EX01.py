import os

import cv2
from matplotlib import pyplot as plt

cwd = os.getcwd()
DATA_PATH = os.path.join(cwd, r'dataset\sequences\00')
orb = cv2.ORB_create()


def read_images(idx: int):
    image_name = "{:06d}.png".format(idx)
    img0 = cv2.imread(DATA_PATH + '\\image_0\\' + image_name, 0)
    img1 = cv2.imread(DATA_PATH + '\\image_1\\' + image_name, 0)
    return img0, img1


# Detect Keypoints and Descriptors
img0_l, img0_r = read_images(0)
kp0_l, desc0_l = orb.detectAndCompute(img0_l, None)
kp0_r, desc0_r = orb.detectAndCompute(img0_r, None)

# Question 1
img0_l_with_kp = cv2.drawKeypoints(img0_l, kp0_l, None, color=(0, 255, 0), flags=0)
img0_r_with_kp = cv2.drawKeypoints(img0_r, kp0_r, None, color=(0, 0, 255), flags=0)

fig = plt.figure(figsize=(16, 9))
fig.add_subplot(1, 2, 1)
plt.imshow(img0_l_with_kp)
plt.axis('off')

fig.add_subplot(1, 2, 2)
plt.imshow(img0_r_with_kp)
plt.axis('off')

plt.savefig(os.path.join(cwd, r'docs\Ex01_q1.jpg'))

# Question 2
for i in range(2):
    print(f"Descriptor {i+1} for L: [" + ','.join(map(str, desc0_l[i])) + ']')
    print(f"Descriptor {i+1} for R: [" + ','.join(map(str, desc0_r[i])) + ']')

# Question 3
plt.cla(), plt.clf()
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = sorted(bf_matcher.match(desc0_l, desc0_r),
                 key=lambda x: x.distance)
img_with_matches = cv2.drawMatches(img0_l, kp0_l, img0_r, kp0_r, matches[:20], None,
                                   matchesThickness=10, flags=0)
plt.imshow(img_with_matches)
plt.axis('off')
plt.show()

# Question 4
# find 2 best matches to calc distance ratio:
matches_2NN = [sorted(match, key=lambda x: x.distance) for match in bf_matcher.knnMatch(desc0_l, desc0_r, 2)]
ratio = 0.5
good, bad = [], []
for first, second in matches_2NN:
    if first.distance / second.distance < ratio:
        good.append(first)
    else:
        bad.append([first, second])

print(f"For a ratio of {ratio}, we kept {len(good)} keypoint matches and discarded {len(bad)}.")
img_with_ratio_filtered_matches = cv2.drawMatches(img0_l, kp0_l, img0_r, kp0_r, good[:20], None,
                                       matchesThickness=10, flags=0)
plt.cla()
plt.imshow(img_with_ratio_filtered_matches)
plt.axis('off')
plt.show()

# find the best match that did not qualify the ratio test:
bad = sorted(bad, key=lambda x: (x[0].distance / x[1].distance, x[0].distance))
false_negative_match = bad[1][0]
img_with_false_negative_match = cv2.drawMatches(img0_l, kp0_l, img0_r, kp0_r, [false_negative_match], None,
                                                matchesThickness=10, flags=0)
plt.cla()
plt.imshow(img_with_false_negative_match)
plt.axis('off')
plt.show()
