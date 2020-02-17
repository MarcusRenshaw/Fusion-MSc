# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:46:03 2017

@author: Knowhow
"""

#
# Compares accuracy of numerical methods
#

# Import all the methods
import euler
import leapfrog
import rk2
import rk4

import matplotlib.pyplot as plt
from numpy import linspace, array, cos, pi, log
from numpy.linalg import norm

# Define the function to solve
def sho(y):
    """ Simple Harmonic Oscillation """
    return array([ y[1], -y[0] ])

y0 = array([1,0])

# number of points
points = array([10,20,40,80,160,320,1000])

# A list of methods to test. Each has a name, a solver function, a step multiplier, and marker
methods = [ ('rk4', rk4.solve, 1, 'o'),
            ('rk2', rk2.solve, 2, '^'),
            ('leapfrog', leapfrog.solve, 4, '+'),
            ('euler', euler.solve, 4, '*') ]

for m in methods:
    # Test each method in turn
    accuracy = []
    for p in points:
        t = linspace(0, 4*pi, p * m[2])  # Multiply the number of steps by a factor
        y = m[1](sho, y0, t)             # Call the solver function
        accuracy.append( norm(y[-1,:] - y0) ) # Error norm
    plt.plot(points, accuracy, label=m[0], marker=m[3])
    order = -log(accuracy[-1] / accuracy[2]) / log(points[-1] / points[2])
    print("Method %s is order %f" % (m[0], order) )

plt.legend(loc='lower left')
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Number of f evaluations')
plt.ylabel('Error in result')
plt.title("Accuracy of explicit methods")
plt.show()
