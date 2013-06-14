#!/usr/bin/env python

import pyximport; pyximport.install()
from jacobian_slam import compute_error_and_jacobian, compute_error
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
import tf
import scipy.sparse as sp
from scikits.sparse.cholmod import cholesky
import cv2


cam_param = 6
point_param = 3

E_element_dtype = [('cam_id', int), ('point_id', int), ('observation_id', int)]

intrinsics = np.array([525., 319.50, 239.5, 0])

slam_data = np.load('slam_final_data.npz')
observations=slam_data['observations']
cameras=slam_data['camera_positions']
points=slam_data['accumulated_keypoints3d']
ground_truth=slam_data['ground_truth']


num_cameras = cameras.shape[1]
num_points = points.shape[1]
num_observations = observations.shape[0]

def fill_hessian(cameras, points, observations):

	B_arr = np.zeros((num_cameras, cam_param, cam_param), dtype=np.float64)
	C_arr = np.zeros((num_points, point_param, point_param), dtype=np.float64)
	E_arr = np.zeros((num_observations, cam_param, point_param), dtype=np.float64)
	E_dict = np.empty(num_observations, dtype=E_element_dtype) 

	v = np.zeros(num_cameras*cam_param, dtype=np.float64).T
	w = np.zeros(num_points*point_param, dtype=np.float64).T
	
	# Compute error, fill B, C, E
	error_sum = 0
	for observation_id in range(num_observations):
		o = observations[observation_id]
		cam_id = o['cam_id']
		point_id = o['point_id']
		cam = cameras[:,cam_id]
		p = points[:,point_id]
		
		R = tf.transformations.quaternion_matrix(cam[4:8])
		e, Jr, Jt, Ji, JX = compute_error_and_jacobian(R[0:3,0:3], cam[1:4], intrinsics, p, o['coord'])
		
		Jc = np.hstack([Jr, Jt])
		

		JctJc = np.dot(Jc.T, Jc)
		JXtJX = np.dot(JX.T, JX)
				
		B_arr[cam_id] += JctJc
		C_arr[point_id] += JXtJX

		E_arr[observation_id] = np.dot(Jc.T, JX)
		E_dict[observation_id] = (cam_id, point_id, observation_id)
	
		v[cam_id*cam_param:(cam_id+1)*cam_param] += np.dot(Jc.T, e)
		w[point_id*point_param:(point_id+1)*point_param] += np.dot(JX.T, e)
	
		error_sum += (e**2).sum()


	E_dict = np.sort(E_dict, order=['cam_id', 'point_id']) 
	E_arr[:] = E_arr[E_dict['observation_id']]
	
	return B_arr, C_arr, E_arr, E_dict, v, w, error_sum


iteration = 0
max_iterations = 50
vv = 2
mu = 0.01

B_arr, C_arr, E_arr, E_dict, v, w, error_sum = fill_hessian(cameras, points, observations)
print 'Inital error', error_sum, 'Inital mean error', error_sum/(num_observations*2)


while iteration < max_iterations:
	iteration += 1

	Cp_inv_arr = np.empty_like(C_arr)
	Bp_arr = np.empty_like(B_arr)

	#compute B+mu*I
	for i in range(num_cameras):
		Bp_arr[i] = B_arr[i] + mu*np.diag(B_arr[i].diagonal())

	#invert C+mu*I
	for i in range(num_points):
		Cp_inv_arr[i] = np.linalg.inv(C_arr[i] + mu*np.diag(C_arr[i].diagonal()))

	indptr = np.arange(num_points+1)
	indices = np.arange(num_points)
	C_inv = sp.bsr_matrix((Cp_inv_arr, indices, indptr), blocksize=(point_param, point_param))

	indptr = np.arange(num_cameras+1)
	indices = np.arange(num_cameras)
	B = sp.bsr_matrix((Bp_arr, indices, indptr), blocksize=(cam_param, cam_param))

	indices = E_dict['point_id']
	u, indptr = np.unique(E_dict['cam_id'], return_index=True)
	indptr = np.hstack([indptr, len(indices)])
	E = sp.bsr_matrix((E_arr, indices, indptr), blocksize=(cam_param, point_param))

	EC_inv = E*C_inv
	S = B + EC_inv*E.T
	k = v - EC_inv*w

	factor = cholesky(S)
	y = factor(k)[:,0]
	z = C_inv * (w - E.T*y)


	camera_new = np.empty_like(cameras)
	points_new = np.empty_like(points)
	
	for i in range(num_cameras):
		cam = cameras[:,i]
		update = y[i*cam_param:(i+1)*cam_param]
		
		R = tf.transformations.quaternion_matrix(cam[4:8])
		dR, J = cv2.Rodrigues(-update[0:3])
		R[0:3,0:3] = np.dot(dR, R[0:3,0:3])
		q = tf.transformations.quaternion_from_matrix(R)
		
		camera_new[0] = cam[0]
		camera_new[1:4,i] = cam[1:4] - update[0:3]
		camera_new[4:8,i] = q
	
	
	for i in range(num_points):
		update = z[i*point_param:(i+1)*point_param]
		points_new[0:3,i] = points[0:3,i] - update
		points_new[3,i] = 1

	new_error_sum = 0
	for observation_id in range(num_observations):
		o = observations[observation_id]
		cam_id = o['cam_id']
		point_id = o['point_id']
		cam = camera_new[:,cam_id]
		p = points_new[:,point_id]
		
		R = tf.transformations.quaternion_matrix(cam[4:8])
		e = compute_error(R[0:3,0:3], cam[1:4], intrinsics, p, o['coord'])
		
		new_error_sum += (e**2).sum()

	F_gain = error_sum - new_error_sum
	L_gain = -(np.dot(y.T,mu*y - v) + np.dot(z.T,mu*z - w))/2
	sigma = F_gain/L_gain
	
	print '********************************************************************'
	print 'Iteration', iteration
	print 'F gain', F_gain, 'L gain', L_gain, 'sigma', sigma, 'mu', mu

	if sigma > 0:
		cameras = camera_new
		points = points_new

		B_arr, C_arr, E_arr, E_dict, v, w, error_sum = fill_hessian(cameras, points, observations)
		print 'Error', error_sum, 'mean error', error_sum/(num_observations*2)
		
		mu = mu*max(1.0/3, 1-(2*sigma-1)**3)
		vv = 2
	else:
		mu = mu * vv
		vv = 2 * vv


'''
print 'Total number of keypoints', accumulated_keypoints3d.shape[1]
print 'Total number of camera positions', camera_positions.shape[1]
print 'Total number of observations', len(observations)
print 'Total number of points with one observation', np.count_nonzero(np.bincount(observations['point_id']) == 1)

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
mlab.plot3d(ground_truth[1], ground_truth[2], ground_truth[3], tube_radius=None, color=(1,0,0))
'''
'''
for i in range(camera_positions.shape[1]):
	
	pos_item = camera_positions[:,i]

	r = tf.transformations.quaternion_matrix(pos_item[4:8])
	r[0:3,3] = pos_item[1:4]

	mlab.quiver3d(r[0,3], r[1,3], r[2,3], r[0,0], r[1,0], r[2,0], color=(1,0,0), mode='2ddash', scale_factor=0.1)
	mlab.quiver3d(r[0,3], r[1,3], r[2,3], r[0,1], r[1,1], r[2,1], color=(0,1,0), mode='2ddash', scale_factor=0.1)
	mlab.quiver3d(r[0,3], r[1,3], r[2,3], r[0,2], r[1,2], r[2,2], color=(0,0,1), mode='2ddash', scale_factor=0.1)
'''
'''
mlab.show()
'''
