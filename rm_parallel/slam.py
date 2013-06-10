#!/usr/bin/env python

import cv2, numpy as np
import itertools
rgbg = cv2.imread("../rgbd_dataset_freiburg1_desk/rgb/1305031453.359684.png")
depth = cv2.imread("../rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png", cv2.CV_LOAD_IMAGE_UNCHANGED)
rgb2g = cv2.imread("../rgbd_dataset_freiburg1_desk/rgb/1305031453.391690.png")
depth2 = cv2.imread("../rgbd_dataset_freiburg1_desk/depth/1305031453.404816.png", cv2.CV_LOAD_IMAGE_UNCHANGED)

rgb = cv2.cvtColor(rgbg, cv2.COLOR_BGR2GRAY)

surfDetector = cv2.FeatureDetector_create("SURF")
surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")

keypoints = surfDetector.detect(rgb, (depth != 0).view(np.uint8))
(keypoints, descriptors) = surfDescriptorExtractor.compute(rgb, keypoints)


flann_params = dict(algorithm=1, trees=4)
flann = cv2.flann_Index(descriptors, flann_params)

cam = np.identity(4)
K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])

rgb2 = cv2.cvtColor(rgb2g, cv2.COLOR_BGR2GRAY)
keypoints2 = surfDetector.detect(rgb2, (depth2 != 0).view(np.uint8))
(keypoints2, descriptors2) = surfDescriptorExtractor.compute(rgb2, keypoints2)

idx, dist = flann.knnSearch(descriptors2, 1, params={})

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

