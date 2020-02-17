from scipy.integrate import odeint
from numpy import linspace, exp, sqrt, pi
import matplotlib.pyplot as plt
import numpy as np

#Function retunring dphi/dx and dE/dx
def dfdx(phiE, x, vs, L):      #parameters passed from odient in result variable
   
    #Unpack values for variables phi and E
    phi = phiE[0] ; E = phiE[1] ; vi = phiE[2]
    
    # Equation for ion velocity
    
    # Equation for ion density    
    ni = (vs/vi) 
    # Equation for electron density
    ne = exp(phi)                
    
    #Calculate the x derivatives
    dEdx = ni - ne
    dphidx = -E
    dvidx = E/vi - vi/L 
    return [dphidx, dEdx, dvidx]

v = [1.0]
C= [0.1,1,10,100,1000,10000]

def run(Vs, L):                   #initiates the index of vs to be passed to odeint
    for Vs in v:
        Array = []             #loop for values values of vs
        for L in C:
            x=linspace(0,40.0,100)    #Define x-axis coordinates
    
            f0 = [0.001, 0.0, Vs]         #Initial values for phi and E
            
            #assigns values from using odeint to variable
            result = odeint(dfdx, f0, x, args=(Vs,L,))    
            
            #phi_of_x = result[:,0]    # Draws out values of phi from result
            phi_of_x = result[:,0]
            vi_of_x = result[:,2]
            
            j = sqrt(1860./ (2.*pi)) * exp(phi_of_x) - 1
            Xwall = np.interp(0.0, j[::-1], x[::-1])
            
            print(Xwall)
            Vwall = x - Xwall
            #Plots current against x
            plt.figure(1)
            plt.xlim([-20,20])
            plt.ylim([0,5])
            plt.plot(Vwall,vi_of_x)
            plt.xlabel("Distance from wall [Debye lengths]")
            plt.ylabel("Current [Normalised]")
            plt.title('Current with respect to the distance with overplotted\
     sheath velocities')
            plt.legend(loc='best')
            plt.grid()
            
            
            #A = np.interp(0.0, Vwall, vi_of_x)
            #Array.append(A)
        plt.grid()
        plt.show()    
    plt.axes([0.2,0.5,0.2,0.3])
    plt.plot(C,Array)
    plt.xscale('log')
    plt.grid()
    plt.show()
if __name__ == "__main__":      #Allows other programs to use the function
    #pass initial conditions,(Phi, Potential, [sheath ion velocities])
    run(v,C)         
                               






