#!/usr/bin/env python

import cv2
import numpy as np
from mayavi import mlab
from os import listdir
from os.path import isfile, join, splitext
import matplotlib.pyplot as plt

# Flann index types. Should be in cv2, but currently they are not there
FLANN_INDEX_KDTREE = 1  
FLANN_INDEX_LSH    = 6
FRAME_COUNT = 25

# Necessary Paths
DEPTH_FOLDER = "../rgbd_dataset_freiburg1_desk/depth"
RGB_FOLDER = "../rgbd_dataset_freiburg1_desk/rgb"

# Initial camera transformation
camera_positions = []

# Intrinsic parameters of the camera
K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])
K_inv = np.linalg.inv(K)

# Keypoint detector and extractor
surfDetector = cv2.FeatureDetector_create("SURF")
surfDetector.setInt('hessianThreshold', 400)
surfDetector.setBool('extended', True)
surfDetector.setBool('upright', True)

surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")


# quaternion to Rotation matrix
def quat2R(tq):
    Rt = np.eye(4)
    Rt[0,0] = 1 - 2*tq[5]*tq[5] - 2*tq[6]*tq[6]
    Rt[0,1] = 2*(tq[4]*tq[5] - tq[6]*tq[7])
    Rt[0,2] = 2*(tq[4]*tq[6] + tq[5]*tq[7])
    Rt[1,0] = 2*(tq[4]*tq[5] + tq[6]*tq[7])
    Rt[1,1] = 1 - 2*tq[4]*tq[4] - 2*tq[6]*tq[6]
    Rt[1,2] = 2*(tq[5]*tq[6] - tq[4]*tq[7])
    Rt[2,0] = 2*(tq[4]*tq[6] - tq[5]*tq[7])
    Rt[2,1] = 2*(tq[4]*tq[7] + tq[5]*tq[6])
    Rt[2,2] = 1 - 2*tq[4]*tq[4] - 2*tq[5]*tq[5]
    Rt[0,3] = tq[1]
    Rt[1,3] = tq[2]
    Rt[2,3] = tq[3]
    return Rt
    
# Search nearest index(difference in timestamps) to a starting point in 
# the scan list
def find_best_start(starting_point, scan_list):
    difference = float('inf')
    best = 0
    for i in range(len(scan_list)):
        if difference > abs(starting_point - scan_list[i][0]):
            difference = abs(starting_point - scan_list[i][0])
            best = i
        if difference < abs(starting_point - scan_list[i][0]):
            break
    return difference, best


# Find Matches between depth and rgb images based on timestamps
def find_sequence(depth_files, rgb_files):
    Matches = []
    rgb_idx = 0
    depth_idx = 0
    while True:
        if rgb_idx >= len(rgb_files)-1 or depth_idx >= len(depth_files)-1:
            break

        diff_rgbs_depth, idx_rgbs_depth = find_best_start(rgb_files[rgb_idx][0], depth_files[depth_idx:])
        diff_depths_rgb, idx_depths_rgb = find_best_start(depth_files[depth_idx][0], rgb_files[rgb_idx:])

        if diff_rgbs_depth > diff_depths_rgb:
            rgb_idx += idx_depths_rgb
            Matches.append((rgb_idx, depth_idx))
        else:
            depth_idx += idx_rgbs_depth
            Matches.append((rgb_idx, depth_idx))
        rgb_idx += 1
        depth_idx += 1
    return Matches


# Computes 2d features, their 3d positions and descriptors
def compute_features(rgb, depth):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    keypoints = surfDetector.detect(gray, (depth != 0).view(np.uint8))
    keypoints, descriptors = surfDescriptorExtractor.compute(gray, keypoints)

    keypoints3d = np.empty((4,len(keypoints)), dtype=np.float64)
    for i in range(len(keypoints)):
        p = keypoints[i].pt
        d = depth[int(p[1]), int(p[0])]/5000.0
        keypoints3d[0,i] = p[0]*d
        keypoints3d[1,i] = p[1]*d
        keypoints3d[2,i] = d
        keypoints3d[3,i] = 1

    keypoints3d[0:3,:] = np.dot(K_inv, keypoints3d[0:3,:])

    return keypoints, keypoints3d, descriptors


# Estimates transform between src and dst points.
# The algorithm is based on:
# "Least-squares estimation of transformation parameters between two point patterns",
# Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
def umeyama(src, dst):
    assert src.shape == dst.shape

    m = src.shape[0]
    n = src.shape[1]
    one_over_n = 1.0/n

    # compute mean
    src_mean = np.sum(src, axis=1) * one_over_n
    dst_mean = np.sum(dst, axis=1) * one_over_n

    # demean
    src_demean = src - src_mean[:,np.newaxis]
    dst_demean = dst - dst_mean[:,np.newaxis]


    # Eq. (36)-(37)
    src_var = np.sum(src_demean**2) * one_over_n

    # Eq. (38)
    sigma = one_over_n * np.dot(dst_demean, src_demean.T);

    U, d, Vt = np.linalg.svd(sigma)

    # Initialize the resulting transformation with an identity matrix...
    Rt = np.eye(m+1)

    # Eq. (39)
    S = np.ones(m);
    if np.linalg.det(sigma) < 0:
        S[m-1] = -1;
    
    # Eq. (40) and (43)
    rank = 0
    for i in range(m):
        if d[i] > 1e-12*d[0]:
            rank +=1

    if rank == m-1:
        if np.linalg.det(U) * np.linalg.det(Vt.T) > 0:
            Rt[0:m,0:m] = np.dot(U, Vt)
        else:
            s = S[m-1]
            S[m-1] = -1
            Rt[0:m,0:m] =  np.dot(np.dot(U, np.diag(S)), Vt)
            S[m-1] = s
    else:
        Rt[0:m,0:m] =  np.dot(np.dot(U, np.diag(S)), Vt)
    
    # Eq. (42)
    c = 1.0/src_var * np.dot(d, S);

    # Eq. (41)
    Rt[0:m,m] = dst_mean
    Rt[0:m,m] -= c*np.dot(Rt[0:m,0:m], src_mean)

    return Rt


def estimate_transform_ransac(src, dst, num_iter, distance2_threshold):
    assert src.shape == dst.shape
    
    max_num_inliers = 0
    inliers = None

    for i in range(num_iter):
        # Select 3 random points
        idx = np.random.permutation(src.shape[1])[0:3]
        # Compute transformation using these 3 points
        Rt = umeyama(src[0:3,idx], dst[0:3,idx])
        # Transform src using computed transformation
        src_transformed = np.dot(Rt, src)
        # Compute number of inliers
        dist2 = np.sum((dst -  src_transformed)**2, axis=0)
        num_inliers = np.count_nonzero(dist2 < distance2_threshold)
        if max_num_inliers < num_inliers:
            max_num_inliers = num_inliers
            inliers =  np.nonzero(dist2 < distance2_threshold)[0]
    
    # Reestimate transformations using all inliers
    Rt = umeyama(src[0:3,inliers], dst[0:3,inliers])
    return Rt, inliers
        
    
depth_files = sorted([ (int(splitext(f)[0].replace(".","")), f) for f in listdir(DEPTH_FOLDER) if isfile(join(DEPTH_FOLDER,f)) ])
rgb_files = sorted([ (int(splitext(f)[0].replace(".","")), f) for f in listdir(RGB_FOLDER) if isfile(join(RGB_FOLDER,f)) ])

sequence = find_sequence(depth_files, rgb_files)

f = open("../rgbd_dataset_freiburg1_desk/groundtruth.txt")
truth = []

for line in f:
    k = line.split()
    truth.append((int(k[0].replace(".","")), float(k[1]), float(k[2]), 
                float(k[3]), float(k[4]), float(k[5]), float(k[6]), 
                float(k[7])))

truth = sorted(truth)

truth_start_idx = find_best_start(rgb_files[sequence[0][0]][0], truth)

camera_positions.append(quat2R(truth[truth_start_idx[1]]))

rgb1 = cv2.imread(join(RGB_FOLDER, rgb_files[sequence[0][0]][1]))
depth1 = cv2.imread(join(DEPTH_FOLDER, depth_files[sequence[0][1]][1]), cv2.CV_LOAD_IMAGE_UNCHANGED)
keypoints1, keypoints3d1, descriptors1 =  compute_features(rgb1, depth1)

observation = []

for seq in range(FRAME_COUNT):
    print seq
    rgb2 = cv2.imread(join(RGB_FOLDER, rgb_files[sequence[seq+1][0]][1]))
    depth2 = cv2.imread(join(DEPTH_FOLDER, depth_files[sequence[seq+1][1]][1]), cv2.CV_LOAD_IMAGE_UNCHANGED)

    keypoints2, keypoints3d2, descriptors2 =  compute_features(rgb2, depth2)

    # Match keypoints
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    flann = cv2.flann_Index(descriptors1, flann_params)
    idx, dist = flann.knnSearch(descriptors2, 1, params={})

    # 3d coordinates of matched points.
    matched_keypoints3d1 = keypoints3d1[:,idx[:,0]]


    # Estimate transform using ransac
    Rt, inliers = estimate_transform_ransac(matched_keypoints3d1, keypoints3d2, 200, 0.01**2)
    camera_positions.append(np.dot(Rt,camera_positions[-1]))
        
    outliers = np.setdiff1d(np.arange(len(keypoints2)), inliers)
    
    for outlier in outliers:
        keypoints1 = np.hstack([keypoints1, keypoints2[outlier]])
        descriptors1 = np.vstack([descriptors1, descriptors2[outlier]])
        keypoints3d1 = np.hstack([keypoints3d1, keypoints3d2[:,[outlier]]])

    for inlier in inliers:
        observation.append((seq, inlier, keypoints2[inlier].pt))
    
'''
print 'Transformation:\n', Rt
print 'Number of inliers: ', inliers.shape[0]

im_k = cv2.drawKeypoints(rgb1, keypoints1)
im2_k = cv2.drawKeypoints(rgb2, keypoints2)

im_keypoints = np.hstack([im_k, im2_k]).copy()

for i in inliers:
    p1 = keypoints1[idx[i]].pt
    p1 = (int(p1[0]), int(p1[1]))
    p2 = keypoints2[i].pt
    p2 = (int(p2[0]+640), int(p2[1]))
    cv2.line(im_keypoints, p1, p2, (0, 255, 0))


cv2.imshow('img',im_keypoints)
cv2.waitKey(0)
'''

# Plot keypoints in 3d
#mlab.points3d(keypoints3d1[0],keypoints3d1[1], keypoints3d1[2], mode='point', color=(1,1,1))


estimate = []
for pos in range(len(camera_positions)):
    estimate.append((rgb_files[sequence[pos][0]][0]/100, 
                    float(camera_positions[pos][0,3]), 
                    float(camera_positions[pos][1,3]), 
                    float(camera_positions[pos][2,3])))

estimate = sorted(estimate)
seq2 = find_sequence(truth, estimate)

tx1 = []
ty1 = []
tz1 = []

tx2 = []
ty2 = []
tz2 = []

for i in seq2:
    tx1.append(estimate[i[0]][1])
    ty1.append(estimate[i[0]][2])
    tz1.append(estimate[i[0]][3])
    tx2.append(truth[i[1]][1])
    ty2.append(truth[i[1]][2])
    tz2.append(truth[i[1]][3])


plt.plot(range(FRAME_COUNT), tx1[:FRAME_COUNT],'b.')
plt.plot(range(FRAME_COUNT), ty1[:FRAME_COUNT],'bo')
plt.plot(range(FRAME_COUNT), tz1[:FRAME_COUNT],'b+')


'''
# Plot camera positions in 3d
for r in camera_positions:
    mlab.quiver3d(r[0,3], r[1,3], r[2,3], r[0,0], r[1,0], r[2,0], color=(1,0,0), mode='2ddash', scale_factor=0.1)
    mlab.quiver3d(r[0,3], r[1,3], r[2,3], r[0,1], r[1,1], r[2,1], color=(0,1,0), mode='2ddash', scale_factor=0.1)
    mlab.quiver3d(r[0,3], r[1,3], r[2,3], r[0,2], r[1,2], r[2,2], color=(0,0,1), mode='2ddash', scale_factor=0.1)'''

plt.plot(range(FRAME_COUNT), tx2[:FRAME_COUNT],'r.')
plt.plot(range(FRAME_COUNT), ty2[:FRAME_COUNT],'ro')
plt.plot(range(FRAME_COUNT), tz2[:FRAME_COUNT],'r+')

#mlab.show()
plt.axis([-1,25,-0.25,2])
plt.show()
