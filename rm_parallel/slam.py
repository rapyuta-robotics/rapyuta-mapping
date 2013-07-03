#!/usr/bin/env python

import cv2
import numpy as np
from mayavi import mlab
from os import listdir
from os.path import isfile, join, splitext
import matplotlib.pyplot as plt
import tf

# Flann index types. Should be in cv2, but currently they are not there
FLANN_INDEX_KDTREE = 1  
FLANN_INDEX_LSH    = 6
FRAME_COUNT = 25

# Necessary Paths
DATASET_FOLDER = "../panorama5"

image_list_dtype =  [('timestamp', float), ('filename', 'S20')]

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

matcher = cv2.BFMatcher(cv2.NORM_L2)


# Computes 2d features, their 3d positions and descriptors
def compute_features(rgb, depth):
	gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

	threshold = 400
	surfDetector.setInt('hessianThreshold', threshold)
	keypoints = surfDetector.detect(gray, (depth != 0).view(np.uint8))

	for i in range(5):
		if len(keypoints) < 300:
			threshold = threshold/2
			surfDetector.setInt('hessianThreshold', threshold)
			keypoints = surfDetector.detect(gray, (depth != 0).view(np.uint8))
		else:
			break

	keypoints = keypoints[0:400]
	keypoints, descriptors = surfDescriptorExtractor.compute(gray, keypoints)

	keypoints3d = np.empty((4,len(keypoints)), dtype=np.float64)
	for i in range(len(keypoints)):
		p = keypoints[i].pt
		d = depth[int(p[1]), int(p[0])]/5000.0
		if d == 0:
			keypoints3d[0,i] = np.nan
			keypoints3d[1,i] = np.nan
			keypoints3d[2,i] = np.nan
			keypoints3d[3,i] = np.nan
		else:
			keypoints3d[0,i] = p[0]*d
			keypoints3d[1,i] = p[1]*d
			keypoints3d[2,i] = d
			keypoints3d[3,i] = 1

	idx = np.isfinite(keypoints3d[0])
	if idx.shape[0] == 0:
		return [], [], []
	keypoints = [keypoints[i] for i in range(idx.shape[0]) if idx[i] == True]
	keypoints3d = keypoints3d[:,idx]
	descriptors = descriptors[idx]

	keypoints3d[0:3,:] = np.dot(K_inv, keypoints3d[0:3,:])

	return keypoints, keypoints3d, descriptors


# Estimates transform between src and dst points.
# The algorithm is based on:
# "Least-squares estimation of transformation parameters between two point patterns",
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


def estimate_transform_ransac(src, dst, num_iter, distance2_threshold, min_num_inliers):
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
    
    if max_num_inliers < min_num_inliers:
		return None, []
    
    # Reestimate transformations using all inliers
    Rt = umeyama(src[0:3,inliers], dst[0:3,inliers])
    print 'Finished ransac with', max_num_inliers, 'inliers'
    return Rt, inliers


def read_image_list(filename):
	f = open(filename)
	fl = []
	for line in f:
		if line[0] == '#':
			continue
		timestamp, filename = line.split()
		fl.append((float(timestamp), filename))
	return np.array(fl, dtype=[('timestamp', float), ('filename', 'S30')])
	
def read_ground_truth(filename):
	f = open(filename)
	fl = []
	for line in f:
		if line[0] == '#':
			continue
		fl.append([float(x) for x in line.split()])
	return np.array(fl, dtype=np.float64).T
	
def find_closest_idx(array, timestamp):
	idx = np.searchsorted(array, timestamp)
	if abs(array[idx] - timestamp) < abs(array[idx-1] - timestamp):
		return idx
	else:
		return idx-1

# read list of files and ground truth
rgb_image_list = read_image_list(join(DATASET_FOLDER, 'rgb.txt'))
depth_image_list = read_image_list(join(DATASET_FOLDER, 'depth.txt'))

has_ground_truth = False
if isfile(join(DATASET_FOLDER, 'groundtruth.txt')):
	ground_truth = read_ground_truth(join(DATASET_FOLDER, 'groundtruth.txt'))
	has_ground_truth = True



rgb_list_item = rgb_image_list[0]
rgb = cv2.imread(join(DATASET_FOLDER, rgb_list_item['filename']))

depth_idx = find_closest_idx(depth_image_list['timestamp'], rgb_list_item['timestamp'])
depth_list_item = depth_image_list[depth_idx]
depth = cv2.imread(join(DATASET_FOLDER, depth_list_item['filename']), cv2.CV_LOAD_IMAGE_UNCHANGED)

if has_ground_truth:
	ground_truth_idx = find_closest_idx(ground_truth[0], rgb_list_item['timestamp'])
	ground_truth_item = ground_truth[:,ground_truth_idx]

	camera_positions = []
	camera_positions.append(ground_truth[:,ground_truth_idx])
	Mwc = tf.transformations.quaternion_matrix(ground_truth_item[4:8])
	Mwc[0:3,3] = ground_truth_item[1:4]
else:
	Mwc = np.eye(4)
	val = np.array([rgb_list_item['timestamp'], 0,0,0,0,0,0,1])
	camera_positions.append(val)

accumulated_keypoints, accumulated_keypoints3d, accumulated_descriptors = compute_features(rgb, depth)
accumulated_keypoints3d = np.dot(Mwc, accumulated_keypoints3d)
accumulated_weights = np.ones(accumulated_descriptors.shape[0])

observations = []



for i in range(len(accumulated_keypoints)):
	observations.append((0, i, accumulated_keypoints[i].pt))


for rgb_list_item in rgb_image_list[1:]:
	rgb = cv2.imread(join(DATASET_FOLDER, rgb_list_item['filename']))

	depth_idx = find_closest_idx(depth_image_list['timestamp'], rgb_list_item['timestamp'])
	depth_list_item = depth_image_list[depth_idx]
	depth = cv2.imread(join(DATASET_FOLDER, depth_list_item['filename']), cv2.CV_LOAD_IMAGE_UNCHANGED)
	
	keypoints, keypoints3d, descriptors = compute_features(rgb, depth)
	
	if len(keypoints) < 3:
		print 'Not enough features... Skipping frame'
		continue
	
	# Match keypoints
	flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	flann = cv2.flann_Index(accumulated_descriptors, flann_params)
	idx, dist = flann.knnSearch(descriptors, 1, params={})

	# 3d coordinates of matched points.
	matched_accumulated_keypoints3d = accumulated_keypoints3d[:,idx[:,0]]

	# Estimate transform using ransac
	Mwc, inliers = estimate_transform_ransac(keypoints3d, matched_accumulated_keypoints3d, 3000, 0.03**2, 20)
	
	#cv2.imshow('Image', rgb)
	#cv2.imshow('Depth', depth)
	#cv2.waitKey(0)
	
	if Mwc == None:
		print 'No transformation found... Skipping frame'
		continue
	
	inliers_accumulated_idx = idx[inliers][:,0]
	w = accumulated_weights[inliers_accumulated_idx][:,np.newaxis]
	accumulated_descriptors[inliers_accumulated_idx] = (w*accumulated_descriptors[inliers_accumulated_idx] + descriptors[inliers])/(w+1)
	accumulated_weights[inliers_accumulated_idx] += 1
    
	outliers = np.setdiff1d(np.arange(len(keypoints)), inliers)

	num_accumulated_keypoints = accumulated_keypoints3d.shape[1]

	accumulated_descriptors = np.vstack([accumulated_descriptors, descriptors[outliers]])
	accumulated_weights = np.hstack([accumulated_weights, np.ones(outliers.shape[0])])
	accumulated_keypoints3d = np.hstack([accumulated_keypoints3d, np.dot(Mwc, keypoints3d[:,outliers])])

	cam_idx = len(camera_positions)
	for i in range(outliers.shape[0]):
		observations.append((cam_idx, num_accumulated_keypoints+i, keypoints[outliers[i]].pt))

	for i in inliers:
		observations.append((cam_idx, idx[i], keypoints[i].pt))


	estimated_poistion = np.zeros(8)
	estimated_poistion[0] = rgb_list_item['timestamp']
	estimated_poistion[1:4] = Mwc[0:3,3]
	Mwc[0:3,3] = 0
	estimated_poistion[4:8] = tf.transformations.quaternion_from_matrix(Mwc)

	camera_positions.append(estimated_poistion)


camera_positions = np.array(camera_positions).T
observations = np.array(observations, dtype=[('cam_id', int), ('point_id', int), ('coord', np.float64, 2)])

if has_ground_truth:
	np.savez('slam_data.npz', observations=observations, camera_positions=camera_positions, accumulated_keypoints3d=accumulated_keypoints3d, ground_truth=ground_truth)
else:
	np.savez('slam_data.npz', observations=observations, camera_positions=camera_positions, accumulated_keypoints3d=accumulated_keypoints3d, ground_truth=[])

print 'Total number of keypoints', accumulated_keypoints3d.shape[1]
print 'Total number of camera positions', camera_positions.shape[1]
print 'Total number of observations', len(observations)
print 'Total number of points with one observation', np.count_nonzero(np.bincount(observations['point_id']) == 1)

if has_ground_truth:
	plt.plot(ground_truth[0], ground_truth[1],'r')
	plt.plot(ground_truth[0], ground_truth[2],'g')
	plt.plot(ground_truth[0], ground_truth[3],'b')

plt.plot(camera_positions[0], camera_positions[1],'r--')
plt.plot(camera_positions[0], camera_positions[2],'g--')
plt.plot(camera_positions[0], camera_positions[3],'b--')

plt.show()

plt.hist(np.bincount(observations['point_id']), 50, normed=1)
plt.show()

mlab.points3d(accumulated_keypoints3d[0], accumulated_keypoints3d[1], accumulated_keypoints3d[2], mode='point', color=(1,1,1))
mlab.plot3d(camera_positions[1], camera_positions[2], camera_positions[3], tube_radius=None, color=(0,1,0))
if has_ground_truth:
	mlab.plot3d(ground_truth[1], ground_truth[2], ground_truth[3], tube_radius=None, color=(1,0,0))

'''
for i in range(camera_positions.shape[1]):
	
	pos_item = camera_positions[:,i]

	r = tf.transformations.quaternion_matrix(pos_item[4:8])
	r[0:3,3] = pos_item[1:4]

	mlab.quiver3d(r[0,3], r[1,3], r[2,3], r[0,0], r[1,0], r[2,0], color=(1,0,0), mode='2ddash', scale_factor=0.1)
	mlab.quiver3d(r[0,3], r[1,3], r[2,3], r[0,1], r[1,1], r[2,1], color=(0,1,0), mode='2ddash', scale_factor=0.1)
	mlab.quiver3d(r[0,3], r[1,3], r[2,3], r[0,2], r[1,2], r[2,2], color=(0,0,1), mode='2ddash', scale_factor=0.1)
'''

mlab.show()

