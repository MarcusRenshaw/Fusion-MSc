# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 12:57:00 2017

@author: Knowhow
"""

#
# Simple Backwards Euler solution of exponential decay
#
# df/dt = a * f
#

import matplotlib.pyplot as plt
from numpy import exp, linspace, zeros

a = -10.
y0 = 10

t = linspace(0,1,40)
plt.plot(t, y0*exp(a*t), label='Exact solution' )

# Calculate using several different time steps
for n in [20, 10, 5]:
    dt = 1./n
    t = linspace(0,1,n, endpoint=False)
    y = zeros([n])
    y[0] = y0
    for i in range(1,n):
        y[i] = y[i-1] / (1. - a*dt)
    plt.plot(t, y, '-o', label='Backwards Euler method, dt='+str(1./n))

plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()
