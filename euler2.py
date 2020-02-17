# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:03:46 2017

@author: Knowhow
"""

# 
# Solves 
# 
#   dy/dt = a y       y = y0 at t = 0
# 
# Re-uses the Euler solver defined in euler.py
#
# B.Dudson, University of York, 2012

import matplotlib.pyplot as plt
from numpy import exp, linspace

# Import the euler function from euler.py
from euler import solve

a = -10.
y0 = [10]

t = linspace(0,1,20)
plt.plot(t, y0*exp(a*t), label='Exact solution' )

# Calculate using several different time steps
for n in [20, 10, 5]:
  t = linspace(0,1,n, endpoint=False)
  y = solve(lambda y:a*y, y0, t)   # lambda defines a small function
  plt.plot(t, y, '-o', label='Euler method, dt='+str(1./n))

plt.legend()
plt.show()
