# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:54:59 2017

@author: Knowhow
"""

from scipy.integrate import odeint
from numpy import linspace, exp, sqrt, pi
import matplotlib.pyplot as plt
import numpy as np



#Function retunring dphi/dx and dE/dx
def dfdx(phiE, x, vs=1):        #parameters passed from odient in result variable
   
    #Unpack values for variables phi and E
    phi = phiE[0] ; E = phiE[1] ; vi = phiE[2]
    
    # Equation for ion velocity
    vi = (sqrt(vs**2 - 2*phi)) 
    # Equation for ion density    
    ni = (vs/vi) 
    # Equation for electron density
    ne = exp(phi)  
    L=10000
    
    #Calculate the x derivatives
    dEdx = ni - ne
    dphidx = -E
    dvidx = (E/vi - vi/L)
    
    return [dphidx, dEdx, dvidx]
    
def interp_x_from_vi(input, x, vi_of_x):       #parameters are (value of j to 
                                        #interpolate, x array, j array)
                                        
#sorts values in x and j arrays. This is necessary for the interpolation.    
    order = vi_of_x.argsort()                 
    vi_of_x = vi_of_x[order]
    x = x[order]
    return np.interp(input, vi_of_x, x) 

def run(E0, phi0,vs):

    v0 = 1
    x=linspace(-20,20, 100)    #Define x-axis coordinates
    
    f0 = [phi0, E0, v0]         #Initial values for phi and E
    
    result = odeint(dfdx, f0, x, args=(vs,))    #assigns values from using odeint to variable
        
    vi_of_x = result[:,2] * -0.1   # Draws out values of phi from result

    interp = interp_x_from_vi(0, x, vi_of_x)  

    print(interp)
    x_wall = x

    #Plots Potential against x
    plt.figure(1)
    plt.plot(x_wall, vi_of_x, label = 'Potential')
    plt.xlabel("Distance [Debye lengths]")
    plt.ylabel("Potential [Normalised]")
    plt.title('Potential with respect to Distance')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    

      
if __name__ == "__main__":      #Allows other programs to use the function
    run(0.001, 0., 1)         #passes initial conditions 
                                #(Electric field, Potential, velocity of sheath ions)





