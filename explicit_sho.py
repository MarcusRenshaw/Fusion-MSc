# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:46:45 2017

@author: Knowhow
"""

#
# Compare explicit integration methods
#

# Import all the methods
import euler
import leapfrog
import rk2
import rk4

import matplotlib.pyplot as plt
from numpy import linspace, array, cos, pi

# Define the function to solve
def sho(y):
    """ Simple Harmonic Oscillation """
    return array([ y[1], -y[0] ])

# Compare methods with the same number of evaluations
nt = 20 

y0 = array([1,0]) # Starting values

# 4th-order Runge-Kutta
t = linspace(0, 4*pi, nt)
y = rk4.solve(sho, y0, t)
plt.plot(t / (2.*pi), y[:, 0], label='RK4')

# 2nd-order Runge Kutta
t = linspace(0, 4*pi, nt*2)  # Double the number of steps
y = rk2.solve(sho, y0, t)
plt.plot(t / (2.*pi), y[:, 0], label='RK2')

# Leapfrog
t = linspace(0, 4*pi, nt*4)  # Each step only one evaluation
y = leapfrog.solve(sho, y0, t)
plt.plot(t / (2.*pi), y[:, 0], label='Leapfrog')

# Euler
t = linspace(0, 4*pi, nt*4)  # Each step only one evaluation
y = euler.solve(sho, y0, t)
plt.plot(t / (2.*pi), y[:, 0], label='Euler')

plt.plot(t / (2.*pi), cos(t), label='cos(t)')
plt.legend(loc='upper left')
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("y")
plt.title("Comparison of explicit methods")
plt.show()

