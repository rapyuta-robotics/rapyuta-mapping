#!/usr/bin/env python

import pyximport; pyximport.install()
from jacobian import compute_error_and_jacobian, compute_error
import numpy as np
import cv2
import scipy.sparse as sp
from scikits.sparse.cholmod import cholesky
from scipy.linalg import expm
from mayavi import mlab

cam_param = 9
point_param = 3

E_element_dtype = [('cam_id', int), ('point_id', int), ('observation_id', int)]
observations_element_dtype = [('cam_id', int), ('point_id', int), ('coord', np.float64, 2)]

f = open('problem-16-22106-pre.txt')
num_cameras, num_points, num_observations = [int(x) for x in f.readline().split()]

print 'Loading file...'
print 'Number of cameras', num_cameras
print 'Number of points', num_points
print 'Number of observations', num_observations



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
		cam = cameras[cam_id]
		p = points[point_id]
		
		R, Jrod = cv2.Rodrigues(cam[0:3])
		e, JR, Jt, Ji, JX = compute_error_and_jacobian(R, cam[3:6], cam[6:9], p, o['coord'])
		
		Jc = np.hstack([np.dot(JR, Jrod.T), Jt, Ji])
		

		JctJc = np.dot(Jc.T, Jc)
		JXtJX = np.dot(JX.T, JX)
				
		B_arr[cam_id] += JctJc + mu*np.diag(JctJc.diagonal())
		C_arr[point_id] += JXtJX + mu*np.diag(JXtJX.diagonal())

		E_arr[observation_id] = np.dot(Jc.T, JX)
		E_dict[observation_id] = (cam_id, point_id, observation_id)
	
		v[cam_id*cam_param:(cam_id+1)*cam_param] += np.dot(Jc.T, e)
		w[point_id*point_param:(point_id+1)*point_param] += np.dot(JX.T, e)
	
		error_sum += (e**2).sum()


	E_dict = np.sort(E_dict, order=['cam_id', 'point_id']) 
	E_arr[:] = E_arr[E_dict['observation_id']]
	
	return B_arr, C_arr, E_arr, E_dict, v, w, error_sum



cameras = np.empty((num_cameras, cam_param), dtype=np.float64)
points = np.empty((num_points, point_param), dtype=np.float64)
observations = np.empty(num_observations, dtype=observations_element_dtype)




for i in range(num_observations):
	cam_id, point_id, x, y = f.readline().split()
	observations[i] = (int(cam_id), int(point_id), np.array([float(x),float(y)]))

for i in range(num_cameras):
	for j in range(cam_param):
		cameras[i,j] = float(f.readline())
        

for i in range(num_points):
	for j in range(point_param):
		points[i,j] = float(f.readline())


mlab.points3d(points[:,0], points[:,1], points[:,2], mode='point', color=(1,1,1))
mlab.show()




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
		Bp_arr[i] = B_arr[i] + mu * np.eye(cam_param)

	#invert C+mu*I
	for i in range(num_points):
		Cp_inv_arr[i] = np.linalg.inv(C_arr[i] + mu * np.eye(point_param))

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
	camera_new.flat = cameras.flat - y
	points_new.flat = points.flat - z

	new_error_sum = 0
	for observation_id in range(num_observations):
		o = observations[observation_id]
		cam_id = o['cam_id']
		point_id = o['point_id']
		cam = camera_new[cam_id]
		p = points_new[point_id]
		
		R, Jrod = cv2.Rodrigues(cam[0:3])
		e = compute_error(R, cam[3:6], cam[6:9], p, o['coord'])
		
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
		



mlab.points3d(points[:,0], points[:,1], points[:,2], mode='point', color=(1,1,1))
mlab.show()


