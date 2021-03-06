
from scipy.integrate import odeint
from numpy import linspace, exp, sqrt, pi
import matplotlib.pyplot as plt

#Function retunring dphi/dx and dE/dx
def dfdx(phiE, x, vs=1):        #parameters passed from odient in result variable
   
    #Unpack values for variables phi and E
    phi = phiE[0] ; E = phiE[1] 
    
    # Equation for ion density
    ni = (vs/(sqrt(vs**2 - 2*phi))) 
    ne = exp(phi)                   
    
    #Calculate the x derivatives
    dEdx = ni - ne
    dphidx = -E
    return [dphidx, dEdx]


def run(E0, phi0, vs):

    x=linspace(0,40,100)    #Define x-axis coordinates
    
    f0 = [phi0, E0]         #Initial values for phi and E
    
    result = odeint(dfdx, f0, x, args=(vs,))    #assigns values from using odeint to variable
        
    x_of_t = result[:,0]    # Draws out values of phi from result

    j = sqrt(1840/(2*pi))*exp(x_of_t) - 1       #the normalised equation for current
    
    #Plots Potential against x
    plt.figure(1)
    plt.plot(x, x_of_t, label = 'Potential')
    plt.xlabel("Distance [Debye lengths]")
    plt.ylabel("Potential [Normalised]")
    plt.title('Potential with respect to Distance')
    plt.legend(loc='best')
    plt.grid()
    
    #Plots current against x
    plt.figure(2)
    plt.plot(x,j, label = 'Current')
    plt.xlabel("Distance [Debye lengths]")
    plt.ylabel("Current [Normalised]")
    plt.title('Current with respect to the distance')
    plt.legend(loc='best')
    plt.grid()
      
if __name__ == "__main__":      #Allows other programs to use the function
    run(0.001, 0., 1.0)         #passes initial conditions 
                                #(Electric field, Potential, velocity of sheath ions)





