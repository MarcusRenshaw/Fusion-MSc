# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:16:09 2017

@author: Knowhow
"""

#
# Demonstrates how to solve a 2D Laplacian equation using
# SciPy's sparse matrix libraries
#
# d2/dx2 + d2/dy2 = 0 
#
# with X boundaries set to 1, and Y boundaries to 0
#
# Equation is discretized using Finite Differences as a matrix equation
#
#  A x = b
#
# where b is zero in the domain and sets the value of the boundaries,
# and x is the solution
# 
# B.Dudson, University of York, 2012
#

try:
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve, bicg
    from numpy import ones, zeros, arange
except:
    print("Couldn't load necessary libraries. Require SciPy and NumPy")
    exit()

# Size of the domain
nx = 100
ny = 100

# Create a sparse matrix
N = nx * ny              # Number of points
A = lil_matrix((N, N))   # Linked list format. Useful for creating matrices

# Function to map from (x,y) location to row/column index
def indx(i,j):
    return i*ny + j
    
# Fixed boundaries, so set everything to the identity matrix
A.setdiag(ones(N))
    
# Loop over the core of the domain
for x in range(1,nx-1):
    for y in range(1,ny-1):
        i = indx(x,y)
        # d2/dx2 + d2/dy2
        A[i, i]           = -4.
        A[i, indx(x+1,y)] = 1.
        A[i, indx(x-1,y)] = 1.
        A[i, indx(x,y+1)] = 1.
        A[i, indx(x,y-1)] = 1.
        
# Convert to Compressed Sparse Row (CSR) format for efficient solving
A = A.tocsr()
    
# Create a vector for b, containing all zeros
b = zeros(N)
    
# Set the values on x=0 and x=nx-1 boundaries to 1
b[indx(0, arange(ny))] = 1.
b[indx(nx-1, arange(ny))] = 1.

# Direct solve sparse matrix
x = spsolve(A, b)
    
# View X as a 2D array. This avoids copying the array
xv = x.view()
xv.shape = (nx, ny)

import matplotlib.pyplot as plt  # Plotting library
import matplotlib.cm as cm       # For the color maps

# Create a filled contour plot
im = plt.imshow(xv, interpolation='bilinear', origin='lower', cmap=cm.hot)
plt.colorbar() # Add a scale
plt.show() # Display interactively

