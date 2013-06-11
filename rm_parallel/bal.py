#!/usr/bin/env python

import pyximport; pyximport.install()
from jacobian import compute_error_and_jacobian
import numpy as np
import cv2
import scipy.sparse as sp
from scikits.sparse.cholmod import cholesky
from scipy.linalg import expm

cam_param = 9
point_param = 3

f = open('problem-16-22106-pre.txt')
num_cameras, num_points, num_observations = [int(x) for x in f.readline().split()]

print 'Loading file...'
print 'Number of cameras', num_cameras
print 'Number of points', num_points
print 'Number of observations', num_observations

G_0 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
G_1 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
G_2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

observations = []
cameras = np.empty((num_cameras, cam_param), dtype=np.float64)
points = np.empty((num_points, point_param), dtype=np.float64)




for i in range(num_observations):
	cam_id, point_id, x, y = f.readline().split()
	observations.append({'cam_id': int(cam_id), 'point_id': int(point_id), 'coord': np.array([float(x),float(y)])})

for i in range(num_cameras):
	for j in range(cam_param):
		cameras[i,j] = float(f.readline())
        

for i in range(num_points):
	for j in range(point_param):
		points[i,j] = float(f.readline())

for i in range(10):

	B_arr = np.zeros((num_cameras, cam_param, cam_param), dtype=np.float64)
	C_arr = np.zeros((num_points, point_param, point_param), dtype=np.float64)
	E_arr = np.zeros((num_observations, cam_param, point_param), dtype=np.float64)
	E_dict = []

	v = np.zeros(num_cameras*cam_param, dtype=np.float64)
	w = np.zeros(num_points*point_param, dtype=np.float64)

	# Compute error, fill B, C, E
	error_sum = 0
	for observation_id in range(num_observations):
		o = observations[observation_id]
		cam_id = o['cam_id']
		point_id = o['point_id']
		cam = cameras[cam_id]
		p = points[point_id]
		R = cv2.Rodrigues(cam[0:3])[0]
		e, J = compute_error_and_jacobian(R, cam[3:6], cam[6:9], p, o['coord'])
		JtJ = np.dot(J.T,J)
		B_arr[cam_id] += JtJ[0:cam_param, 0:cam_param]
		C_arr[point_id] += JtJ[cam_param:cam_param+point_param, cam_param:cam_param+point_param]

		E_arr[observation_id] = JtJ[0:cam_param, cam_param:cam_param+point_param]
		E_dict.append((cam_id, point_id, observation_id))
	
		Jte = np.dot(J.T,e)
		v[cam_id*cam_param:(cam_id+1)*cam_param] += Jte[0:cam_param]
		w[point_id*point_param:(point_id+1)*point_param] += Jte[cam_param:cam_param+point_param]
	
		error_sum += e.sum()


	print error_sum

	#invert C
	for i in range(num_points):
		C_arr[i] = np.linalg.inv(C_arr[i])

	indptr = np.arange(num_points+1)
	indices = np.arange(num_points)
	C_inv = sp.bsr_matrix((C_arr, indices, indptr), blocksize=(point_param, point_param))

	indptr = np.arange(num_cameras+1)
	indices = np.arange(num_cameras)
	B = sp.bsr_matrix((B_arr, indices, indptr), blocksize=(cam_param, cam_param))

	E_dict = sorted(E_dict)
	E_arr[:] = E_arr[[x[2] for x in E_dict]]
	indices = [x[1] for x in E_dict]
	u, indptr = np.unique([x[0] for x in E_dict], return_index=True)
	indptr = np.hstack([indptr, len(indices)])
	E = sp.bsr_matrix((E_arr, indices, indptr), blocksize=(cam_param, point_param))

	EC_inv = E*C_inv
	S = B + EC_inv*E.T
	k = v - EC_inv*w

	factor = cholesky(S)
	y = factor(k)
	z = C_inv * (w - (E.T*y).T).T
	
	for cam_id in range(num_cameras):
		cam_update = y[cam_id*cam_param:(cam_id+1)*cam_param]
		cam = cameras[cam_id]
		R = cv2.Rodrigues(cam[0:3])[0]
		dR = expm(G_0 * cam_update[0] + G_1 * cam_update[1] + G_2 * cam_update[2])
		cam[0:3] = cv2.Rodrigues(np.dot(dR, R))[0].T
		cam[3:9] = cam[3:9] + cam_update[3:9].T
		cameras[cam_id] = cam

	for point_id in range(num_points):
		point_update = z[point_id*point_param:(point_id+1)*point_param]
		points[point_id] += point_update[0]
		
		

	

