#!/usr/bin/env python

import cv2, numpy as np
import itertools
import random as rnd

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)
    
    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = transpose(AA) * BB

    U, S, Vt = linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
       print "Reflection detected"
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    print t

    return R, t


cam = np.identity(4)
K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])
K_inv = np.linalg.inv(K)


rgbg = cv2.imread("../rgbd_dataset_freiburg1_desk/rgb/1305031453.359684.png")
depth = cv2.imread("../rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png", cv2.CV_LOAD_IMAGE_UNCHANGED)
rgb2g = cv2.imread("../rgbd_dataset_freiburg1_desk/rgb/1305031453.391690.png")
depth2 = cv2.imread("../rgbd_dataset_freiburg1_desk/depth/1305031453.404816.png", cv2.CV_LOAD_IMAGE_UNCHANGED)

rgb = cv2.cvtColor(rgbg, cv2.COLOR_BGR2GRAY)

surfDetector = cv2.FeatureDetector_create("SURF")
surfDetector.setInt('hessianThreshold', 2000)
surfDetector.setBool('extended', True)
surfDetector.setBool('upright', True)

surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")

keypoints = surfDetector.detect(rgb, (depth != 0).view(np.uint8))
(keypoints, descriptors) = surfDescriptorExtractor.compute(rgb, keypoints)

keypoints3d = np.empty((3,len(keypoints)), dtype=np.float32)
for i in range(len(keypoints)):
    p = keypoints[i].pt
    d = depth[int(p[1]), int(p[0])]/5000.0
    keypoints3d[0,i] = p[0]*d
    keypoints3d[1,i] = p[1]*d
    keypoints3d[2,i] = d

keypoints3d = np.dot(K_inv, keypoints3d)




flann_params = dict(algorithm=1, trees=4)
flann = cv2.flann_Index(descriptors, flann_params)


rgb2 = cv2.cvtColor(rgb2g, cv2.COLOR_BGR2GRAY)
keypoints2 = surfDetector.detect(rgb2, (depth2 != 0).view(np.uint8))
(keypoints2, descriptors2) = surfDescriptorExtractor.compute(rgb2, keypoints2)

B = np.empty((3,len(keypoints2)), dtype=np.float32)
for i in range(len(keypoints2)):
    p = keypoints2[i].pt
    d = depth[int(p[1]), int(p[0])]/5000.0
    B[0,i] = p[0]*d
    B[1,i] = p[1]*d
    B[2,i] = d

B = np.dot(K_inv, B)

idx, dist = flann.knnSearch(descriptors2, 1, params={})

iter_count = 20
#for i in xrange(iter_count):
random_idx = rnd.sample(xrange(len(keypoints)), 3)

#A3 = [A[:,i] for i in random_idx]
#B3 = [B[:,i] for i in random_idx]

#ret_R, ret_t = rigid_transform_3D(A, B)

im_k = cv2.drawKeypoints(rgbg, keypoints)
im2_k = cv2.drawKeypoints(rgb2g, keypoints2)

im_keypoints = np.hstack([im_k, im2_k]).copy()


for i in range(len(idx)):
    p1 = keypoints[idx[i]].pt
    p1 = (int(p1[0]), int(p1[1]))
    p2 = keypoints2[i].pt
    p2 = (int(p2[0]+640), int(p2[1]))
    cv2.line(im_keypoints, p1, p2, (0, 255, 0))


cv2.imshow('img',im_keypoints)
cv2.waitKey(0)

