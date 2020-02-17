# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 00:29:45 2017

@author: Knowhow
"""

#
# Central differencing for advection problems on periodic mesh
#

from numpy import concatenate

def central(y, t, dx, v):
    """ 2nd order central differencing on periodic uniform mesh """
    dydx = concatenate( ([y[1]-y[-1]], y[2:] - y[:-2], [y[0]-y[-2]]) ) / (2.*dx)
    return -v * dydx

from numpy import exp, linspace, zeros
from scipy.integrate import odeint # SciPy ODE integration
import matplotlib.pyplot as plt

nx = 100
v = 1.0

x = linspace(0,1,nx, endpoint=False) # Periodic box of length 1
dx = x[1] - x[0]

y0 = exp( -((x - 0.3) / 0.1)**2 )  # Gaussian shape
#y0 = zeros([nx])
y0[int(6*nx/8):int(7*nx/8)] = 1.0   # Top-hat function

tarray = [0,0.1,1]
y = odeint(central, y0, tarray, args = (dx,v))
for i,t in enumerate(tarray):
    plt.plot(x, y[i,:], label="t = %.2f" % (t))

plt.legend(loc='upper center')
plt.show()
