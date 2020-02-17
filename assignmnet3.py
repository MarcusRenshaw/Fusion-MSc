# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:59:43 2017

@author: Knowhow
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 23:14:22 2017

@author: Knowhow
"""

from scipy.integrate import odeint
from numpy import linspace, exp, sqrt, pi
import matplotlib.pyplot as plt
import numpy as np


#Function retunring dphi/dx and dE/dx
def dfdx(phiE, x, vs=1):      #parameters passed from odient in result variable
   
    #Unpack values for variables phi and E
    phi = phiE[0] ; E = phiE[1] 
    
    # Equation for ion velocity
    vi = (sqrt(vs**2 - 2*phi)) 
    # Equation for ion density    
    ni = (vs/vi) 
    # Equation for electron density
    ne = exp(phi)                
    
    #Calculate the x derivatives
    dEdx = ni - ne
    dphidx = -E
    return [dphidx, dEdx]

#Interpolates a result for x at given j
def interp_x_from_j(input, x, j):       #parameters are (value of j to 
                                        #interpolate, x array, j array)
                                        
#sorts values in x and j arrays. This is necessary for the interpolation.    
    order = j.argsort()                 
    j = j[order]
    x = x[order]
    return np.interp(input, j, x)       #returns value of x at given j
    

def run(E0, phi0, vs):
    a = 0                    #initiates the index of vs to be passed to odeint
    for i in vs:             #loop for values values of vs
        
        x=linspace(0,40,100)    #Define x-axis coordinates
    
        f0 = [phi0, E0]         #Initial values for phi and E
        
        #assigns values from using odeint to variable
        result = odeint(dfdx, f0, x, args=(vs[a],))    
        
        phi_of_x = result[:,0]    # Draws out values of phi from result

        ion_electron_mass_ratio = 1840          # defines a ratio for ion-electron mass
        j = sqrt(ion_electron_mass_ratio/(2*pi))*exp(phi_of_x) - 1       #the normalised equation for current
         
    
        #calls the interpolation function with arguments ()
        interp = interp_x_from_j(0, x, j)  
        #makes new variable that is the difference between x and the
        #interpolated value of x at j=0    
        x_wall = x - interp                           
        a += 1
        
        #Plots current against x
        plt.figure(2)
        plt.plot(x_wall,j, label = 'Vs: %s' %(vs[a-1]))
        plt.xlabel("Distance from wall [Debye lengths]")
        plt.ylabel("Current [Normalised]")
        plt.title('Current with respect to the distance with overplotted\
 sheath velocities')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
      
if __name__ == "__main__":      #Allows other programs to use the function
    #pass initial conditions,(Phi, Potential, [sheath ion velocities])
    run(0.001, 0., [1, 1.5, 2,])         
                               






