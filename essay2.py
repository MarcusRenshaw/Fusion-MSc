# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 23:02:30 2017

@author: Knowhow
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:08:49 2017

@author: Knowhow
"""

"""
Debye sheath model
"""

from scipy.integrate import odeint # SciPy ODE integration
from numpy import linspace, exp, sqrt, pi, interp

import matplotlib.pyplot as plt  # Plotting library

def sheath(f, t, Vs):
    # f is an array of all evolving variables
    E = f[0]
    phi = f[1]
    
    ne = exp(phi)                # Boltzman relation
    Vi = sqrt( Vs**2 - 2.*phi )  # Energy conservation
    ni = Vs / Vi                 # Ion continuity
    
    # Calculate the time derivatives
    dEdx = ni - ne                # Poisson's law
    dphidx = -E                   # Electric field
    
    # Return the time derivatives in the same order as in the input f
    return [dEdx, dphidx]

def solve(x=None, Vs=1.0):
    
    x = linspace(0, 40., 100) # Create one
    
    y0 = [0.001, 0.0] # Starting values for [E, phi]
    y = odeint(sheath, y0, x, args = (Vs,))   # Note the comma in "(Vs,)" to make it a tuple
    
    phi = y[:,1]
    j = sqrt(1860./ (2.*pi)) * exp(phi) - 1
    
    return phi, j  # Return both phi and j

if __name__ == "__main__":
    
    x = linspace(0, 40., 100)
    for v in [1]:
        # Solve using this value of v
        phi, j = solve(x, v)
        # Find where j goes to zero
        Xwall = interp(0.0, j[::-1], x[::-1])  # Need to reverse arrays as j is decreasing
        
        # Plot current relative to the wall position
        plt.plot(x, j, label="Vs = "+str(v))
        
    plt.grid(True)  # Add a background grid
    plt.xlabel("x [Debye lengths]")
    plt.ylabel("Current [Normalised]")
    plt.legend()
    
    plt.show()
    