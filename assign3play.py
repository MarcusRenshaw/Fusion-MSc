

from scipy.integrate import odeint # SciPy ODE integration
from numpy import linspace, exp, sqrt, pi, interp

import matplotlib.pyplot as plt  # Plotting library

def sheath(f, t, vs, L):
    # f is an array of all evolving variables
    E = f[0]
    phi = f[1]
    Vi = f[2]
    
    ne = exp(phi)                # Boltzman relation
    Vi = (sqrt(vs**2 - 2*phi))   # Energy conservation
    ni = vs/Vi             # Ion continuity
    
    # Calculate the time derivatives
    dEdx = ni - ne                # Poisson's law
    dphidx = -E                   # Electric field
    dvidx = E/Vi - Vi/L 
    # Return the time derivatives in the same order as in the input f
    return [dEdx, dphidx, dvidx]

def solve(x, L = 1.0, Vs=1.0):
    
   # Create one
    
    y0 = [0.001, 0.0, 1] # Starting values for [E, phi]
    y = odeint(sheath, y0, x, args = (Vs, L))   # Note the comma in "(Vs,)" to make it a tuple
    
    vi = y[:,2]
    
    
    return vi # Return both phi and j

if __name__ == "__main__":
    
   
    x = linspace(-20,40, 100)
        # Solve using this value of v
    f = solve(x, 10, 1.0)
        # Find where j goes to zero
    
    print (f)    
        # Plot current relative to the wall position
    plt.plot(x, f)
        
    plt.grid(True)
    plt.xlim( -25,25)
    plt.ylim( 0, 5 )# Add a background grid
    plt.xlabel("x [Debye lengths]")
    plt.ylabel("Current [Normalised]")
    plt.legend()
    
    plt.show()
    
