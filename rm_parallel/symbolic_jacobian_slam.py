
from sympy import *
from sympy.utilities.iterables import flatten


# Jacobian of Rodrigues vector at (0,0,0)
Jrod = Matrix([
       [ 0.,  0.,  0.],
       [ 0.,  0., -1.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 0.,  0.,  0.],
       [-1.,  0.,  0.],
       [ 0., -1.,  0.],
       [ 1.,  0.,  0.],
       [ 0.,  0.,  0.]])


dR = MatrixSymbol('dR', 3,3)
R = MatrixSymbol('R', 3,3)
X = MatrixSymbol('X', 3,1)
t = MatrixSymbol('t', 3,1)

f, c1, c2, k1 = symbols("f, c1, c2, k1")
u, v = symbols("u, v")


P = Matrix(dR*R*X + t)
p = Matrix([P[0]/P[2], P[1]/P[2]])
p_norm2 = p[0]**2 + p[1]**2
p_prime = f * (1 + k1 * p_norm2) * p

e = Matrix([u - (p_prime[0] + c1), v - (p_prime[1] + c2)])

JdR = e.jacobian(flatten(Matrix(dR)))
JdR = JdR.subs({'dR_00':1, 'dR_01':0, 'dR_02':0, 'dR_10':0, 'dR_11':1, 'dR_12':0, 'dR_20':0, 'dR_21':0, 'dR_22':1})
Jr = JdR*Jrod

e = e.subs({'dR_00':1, 'dR_01':0, 'dR_02':0, 'dR_10':0, 'dR_11':1, 'dR_12':0, 'dR_20':0, 'dR_21':0, 'dR_22':1})

Jt = e.jacobian(flatten(Matrix(t)))
JX = e.jacobian(flatten(Matrix(X)))
Ji = e.jacobian([f, c1, c2, k1])


print """
from cython.view cimport array
import numpy as np

def compute_error_and_jacobian(R, t, intrinsics, X, coord):

	cdef double R_00 = R[0,0]
	cdef double R_01 = R[0,1]
	cdef double R_02 = R[0,2]
	cdef double R_10 = R[1,0]
	cdef double R_11 = R[1,1]
	cdef double R_12 = R[1,2]
	cdef double R_20 = R[2,0]
	cdef double R_21 = R[2,1]
	cdef double R_22 = R[2,2]

	cdef double t_00 = t[0]
	cdef double t_10 = t[1]
	cdef double t_20 = t[2]

	cdef double X_00 = X[0]
	cdef double X_10 = X[1]
	cdef double X_20 = X[2]

	cdef double f = intrinsics[0]
	cdef double c1 = intrinsics[1]
	cdef double c2 = intrinsics[2]
	cdef double k1 = intrinsics[3]

	cdef double u = coord[0]
	cdef double v = coord[1]

	Jr_np = np.empty((2,3), dtype=np.float64)
	cdef double [:, :] Jr = Jr_np

	Jt_np = np.empty((2,3), dtype=np.float64)
	cdef double [:, :] Jt = Jt_np

	Ji_np = np.empty((2,4), dtype=np.float64)
	cdef double [:, :] Ji = Ji_np

	JX_np = np.empty((2,3), dtype=np.float64)
	cdef double [:, :] JX = JX_np

	e_np = np.empty(2, dtype=np.float64)
	cdef double [:] e = e_np
"""

print '\te['+str(0)+'] = ' + str(e[0])
print '\te['+str(1)+'] = ' + str(e[1])


for i in range(Jr.shape[0]):
	for j in range(Jr.shape[1]):
		print '\tJr['+str(i)+','+str(j)+'] = ' + str(Jr[i,j])

for i in range(Jt.shape[0]):
	for j in range(Jt.shape[1]):
		print '\tJt['+str(i)+','+str(j)+'] = ' + str(Jt[i,j])

for i in range(Ji.shape[0]):
	for j in range(Ji.shape[1]):
		print '\tJi['+str(i)+','+str(j)+'] = ' + str(Ji[i,j])

for i in range(JX.shape[0]):
	for j in range(JX.shape[1]):
		print '\tJX['+str(i)+','+str(j)+'] = ' + str(JX[i,j])


print "\treturn e_np, Jr_np, Jt_np, Ji_np, JX_np"


print """

def compute_error(R, t, intrinsics, X, coord):
	cdef double R_00 = R[0,0]
	cdef double R_01 = R[0,1]
	cdef double R_02 = R[0,2]
	cdef double R_10 = R[1,0]
	cdef double R_11 = R[1,1]
	cdef double R_12 = R[1,2]
	cdef double R_20 = R[2,0]
	cdef double R_21 = R[2,1]
	cdef double R_22 = R[2,2]

	cdef double t_00 = t[0]
	cdef double t_10 = t[1]
	cdef double t_20 = t[2]

	cdef double X_00 = X[0]
	cdef double X_10 = X[1]
	cdef double X_20 = X[2]

	cdef double f = intrinsics[0]
	cdef double c1 = intrinsics[1]
	cdef double c2 = intrinsics[2]
	cdef double k1 = intrinsics[3]

	cdef double u = coord[0]
	cdef double v = coord[1]


	e_np = np.empty(2, dtype=np.float64)
	cdef double [:] e = e_np
"""

print '\te['+str(0)+'] = ' + str(e[0])
print '\te['+str(1)+'] = ' + str(e[1])


print "\treturn e_np"

