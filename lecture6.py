# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 23:41:05 2017

@author: Knowhow
"""


from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from numpy import zeros

def solve(N):
	length = 1.0

	dx = length / (N-1)
	
	x = zeros(N)
	for i in range(1,N):
		x[i] = x[i-1] + dx
	# x = [0, dx, 2*dx, 3*dx, ...]		

	A = lil_matrix( (N, N) )

	## Set middle part of the domain (all except boundaries)

	for i in range(1, N-1):
		A[i,i-1] = 1.0 / dx**2 + 1.0/(2.*dx)
		A[i,i]   = -2.0 / dx**2
		A[i,i+1] = 1.0 / dx**2 - 1.0/(2.*dx)

	rhs = zeros(N) - 3

	# Set boundary conditions
	# Fixed value (Dirichlet)

	A[0,0] = 1.0
	A[N-1,N-1] = 1.0
	rhs[0] = 0.0   # Value of the left hand side
	rhs[N-1] = 0.0 # Value of the right hand side

	# Convert A to CSR format
	A = A.tocsr()

	solution = spsolve(A, rhs)
	return x, solution

import matplotlib.pyplot as plt

for i in [3,5,10,20,40]:
	x, sol = solve(i)
	plt.plot(x, sol)

plt.show()
