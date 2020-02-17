# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:39:04 2017

@author: cshepherd
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:36:11 2017

@author: cs1819
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:25:41 2017

@author: cs1819
"""

import matplotlib.pyplot as plt  # Plotting library

from scipy.integrate import odeint # SciPy ODE integration
from numpy import linspace, exp, sqrt, pi, interp
import numpy as numpy

"""
V_s = 1
"""

def dPdx(curP, curX, v_s, L):
    # curP is an array of all evolving variables
    
    phi = curP[0] ; E = curP[1] ; v_i = curP[2]
       
  
    # Calculate the x derivatives of the Poisson's system
    dphidx = -E
    dEdx = (v_s/(sqrt(v_s**2 - 2*phi)) - (exp(phi)))
    dv_idx = E/v_i - v_i/L
    #dEdx = n_i - n_e
    # Return the x derivatives in the same order as in the input   
    return [dphidx, dEdx, dv_idx]


#set initial conditions
phi0 = 0
E0 = 0.001
v_i0 = 1
f0 = [phi0, E0, v_i0]
    
#set parameters
v_s = 1
Lset = [0.1,1,10,100,1000,10000]
L=[0.1,1,10,100,1000,10000]
    
    
x = linspace(0,40,100) #define x positions where we want to know the result - up to x=40, 100pts.
#solving the equation
y = odeint(dPdx, f0, x, args = (v_s,L[0]))
y1 = odeint(dPdx, f0, x, args = (v_s,L[1]))
y10 = odeint(dPdx, f0, x, args = (v_s,L[2]))
y100 = odeint(dPdx, f0, x, args = (v_s,L[3]))
y1000 = odeint(dPdx, f0, x, args = (v_s,L[4]))
y10000 = odeint(dPdx, f0, x, args = (v_s,L[5]))
#y = odeint(dPdx, f0, x)

phi_of_x1 = y[:,0]
E_of_x = y[:,1]
v_i_of_x = y[:,2]   
v_i_of_x1 = y1[:,2] 
v_i_of_x10 = y10[:,2] 
v_i_of_x100 = y100[:,2] 
v_i_of_x1000 = y1000[:,2] 
v_i_of_x10000 = y10000[:,2] 


j1 = sqrt(1840/(2*pi))*exp(phi_of_x1) - 1
#normalised current equation which goes to -1 when phi is very negative.
#here the result from line 51 is inputted into the exp in order to show current vs x in debye lengths

j1inc = numpy.flipud(j1)
#this line flips the j array so it is always increasing. 
XWall1 = interp(0, j1inc, x)
#I have flipped the arguments to find y when x = 0
XWall1real=40-XWall1
#Because the array was flipped we need this line to calculate the true value of x when j =0
#print(XWall1real)


"""
plt.plot(x-XWall1real, v_i_of_x)
plt.plot(x-XWall1real, v_i_of_x1)
plt.plot(x-XWall1real, v_i_of_x10)
plt.plot(x-XWall1real, v_i_of_x100)
plt.plot(x-XWall1real, v_i_of_x1000)
plt.plot(x-XWall1real, v_i_of_x10000)
"""

plt.plot(x, v_i_of_x)
plt.plot(x, v_i_of_x1)
plt.plot(x, v_i_of_x10)
plt.plot(x, v_i_of_x100)
plt.plot(x, v_i_of_x1000)
plt.plot(x, v_i_of_x10000)


""" The above shows that Xwall isn't doing anything. Need to sort that out.
 The shape of the curves is in agreement with the final result needed...
 It looks like there is a huge chunk spliced from the final figure???
"""

plt.show()

"""
"""
#V_s = 1.5
"""

def dPdx(curP, curX, v_s):
    # curP is an array of all evolving variables
    
    phi = curP[0] ; E = curP[1]
       
  
    # Calculate the x derivatives of the Poisson's system
    dphidx = -E
    dEdx = (v_s/(sqrt(v_s**2 - 2*phi)) - (exp(phi)))
    #dEdx = n_i - n_e
    # Return the x derivatives in the same order as in the input   
    return [dphidx, dEdx]


#set initial conditions
phi0 = 0
E0 = 0.001
f0 = [phi0, E0]
    
#set parameters
v_s15 = 1.5
    
    
x = linspace(0,40,100) #define x positions where we want to know the result - up to x=40, 100pts.
#solving the equation
y = odeint(dPdx, f0, x, args = (v_s15,))
#y = odeint(dPdx, f0, x)

phi_of_x2 = y[:,0]
E_of_x = y[:,1]


j2 = sqrt(1840/(2*pi))*exp(phi_of_x2) - 1



"""
#V_s = 2
"""

def dPdx(curP, curX, v_s):
    # curP is an array of all evolving variables
    
    phi = curP[0] ; E = curP[1]
       
  
    # Calculate the x derivatives of the Poisson's system
    dphidx = -E
    dEdx = (v_s/(sqrt(v_s**2 - 2*phi)) - (exp(phi)))
    #dEdx = n_i - n_e
    # Return the x derivatives in the same order as in the input   
    return [dphidx, dEdx]


#set initial conditions
phi0 = 0
E0 = 0.001
f0 = [phi0, E0]
    
#set parameters
v_s2 = 2
    
    
x = linspace(0,40,100) #define x positions where we want to know the result - up to x=40, 100pts.
#solving the equation
y = odeint(dPdx, f0, x, args = (v_s2,))
#y = odeint(dPdx, f0, x)

phi_of_x3 = y[:,0]
E_of_x = y[:,1]


j3 = sqrt(1840/(2*pi))*exp(phi_of_x3) - 1


#j1 corresponds to v_s =1, j2 = v_s =1.5, j3 = v_s=2

"""
#Now we have our current values for each v_s value we can find the value of x where j = 0 and overplot
#them onto one graph

"""

j1inc = numpy.flipud(j1)
#this line flips the j array so it is always increasing. 
XWall1 = interp(0, j1inc, x)
#I have flipped the arguments to find y when x = 0
XWall1real=40-XWall1
#Because the array was flipped we need this line to calculate the true value of x when j =0
#print(XWall1real)

j2inc = numpy.flipud(j2)
XWall2 = interp(0, j2inc, x)
XWall2real=40-XWall2
#print(XWall2real)
    
j3inc = numpy.flipud(j3)
XWall3 = interp(0, j3inc, x)
XWall3real=40-XWall3
#print(XWall3real)

plt.plot((x-XWall1real),j1, label = 'V_s = 1')

plt.plot((x-XWall2real),j2, label = 'V_s = 1.5')

plt.plot((x-XWall3real),j3, label = 'V_s = 2')

plt.legend()

plt.xlabel("x [Debye lengths]")
plt.ylabel("Current [Normalised]")
plt.grid()
plt.xlim( (-40, 30) )
plt.ylim( (-5, 20) )


plt.title("")
plt.show()
"""