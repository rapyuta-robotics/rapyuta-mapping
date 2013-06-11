
from sympy import *
from sympy.utilities.iterables import flatten

G_0 = Matrix([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
G_1 = Matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
G_2 = Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
Jg = Matrix([flatten(G_0), flatten(G_1), flatten(G_2)]).T

R = MatrixSymbol('R', 3,3)
X = MatrixSymbol('X', 3,1)
t = MatrixSymbol('t', 3,1)

f, k1, k2 = symbols("f, k1, k2")
u, v = symbols("u, v")


P = Matrix(R*X + t)
p = Matrix([-P[0]/P[2], -P[1]/P[2]])
p_norm2 = p[0]**2 + p[1]**2
p_norm4 = p_norm2**2
p_prime = f * (1 + k1 * p_norm2 + k2 * p_norm4) * p

e = Matrix([(u - p_prime[0])**2, (v -  p_prime[1])**2])

JR = e.jacobian(flatten(Matrix(R)))
Jr = JR*Jg

Jt = e.jacobian(flatten(Matrix(t)))
JX = e.jacobian(flatten(Matrix(X)))
Jfk1k2 = e.jacobian([f, k1, k2])


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
	cdef double k1 = intrinsics[1]
	cdef double k2 = intrinsics[2]

	cdef double u = coord[0]
	cdef double v = coord[1]

	J_np = np.empty((2,12), dtype=np.float64)
	cdef double [:, :] J = J_np

	e_np = np.empty(2, dtype=np.float64)
	cdef double [:] e = e_np
"""

print '\te['+str(0)+'] = ' + str(e[0])
print '\te['+str(1)+'] = ' + str(e[1])


for i in range(Jr.shape[0]):
	for j in range(Jr.shape[1]):
		print '\tJ['+str(i)+','+str(j)+'] = ' + str(Jr[i,j])

for i in range(Jt.shape[0]):
	for j in range(Jt.shape[1]):
		print '\tJ['+str(i)+','+str(j+3)+'] = ' + str(Jt[i,j])

for i in range(Jfk1k2.shape[0]):
	for j in range(Jfk1k2.shape[1]):
		print '\tJ['+str(i)+','+str(j+6)+'] = ' + str(Jfk1k2[i,j])

for i in range(JX.shape[0]):
	for j in range(JX.shape[1]):
		print '\tJ['+str(i)+','+str(j+9)+'] = ' + str(JX[i,j])


print "\treturn e_np, J_np"



