# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 00:32:25 2017

@author: Knowhow
"""

#
# Central differencing for advection problems on 
# periodic mesh using Euler time integration
#

from numpy import concatenate

def central(y, t, dx, v):
    """ 2nd order central differencing on periodic uniform mesh """
    dydx = concatenate( ([y[1]-y[-1]], y[2:] - y[:-2], [y[0]-y[-2]]) ) / (2.*dx)
    return -v * dydx

from numpy import exp, linspace, zeros
import matplotlib.pyplot as plt

nx = 100
v = 1.0

x = linspace(0,1,nx, endpoint=False) # Periodic box of length 100
dx = x[1] - x[0]

y0 = exp( -((x - 0.3) / 0.1)**2 )  # Gaussian shape ()
y0[int(6*nx/8):int(7*nx/8)] = 1.0   # Top-hat function

plt.plot(x,y0)

#Time settings
tend=1.0
nt = 11
time=linspace(0,tend,nt)
dt=time[1]

#Initial conditions
yy=zeros([nt,nx])
yy[0,:]=y0

#Integrate in time with Euler
for j in range(nt-1):
    yy[j+1,:]=yy[j,:]+dt*central(yy[j,:],time[j],dx,v)
    
    
"""
plt.plot(x,yy[0,:],label="t = %.2f" % (time[0]))
plt.plot(x,yy[1,:],label="t = %.2f" % (time[1]))
plt.plot(x,yy[nt-1,:],label="t = %.2f" % (time[nt-1]))

plt.legend(loc='upper center')
plt.show()
"""