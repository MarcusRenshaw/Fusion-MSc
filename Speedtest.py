#!/usr/bin/env python
#
# Electrostatic PIC code in a 1D cyclic domain

from numpy import arange, concatenate, zeros, linspace, floor, array, pi
from numpy import sin, cos, sqrt, random, histogram, where, sum
import numpy as np
import time

import matplotlib.pyplot as plt # Matplotlib plotting library

try:
    import matplotlib.gridspec as gridspec  # For plot layout grid
    got_gridspec = True
except:
    got_gridspec = False

# Need an FFT routine, either from SciPy or NumPy
try:
    from scipy.fftpack import fft, ifft
except:
    # No SciPy FFT routine. Import NumPy routine instead
    from numpy.fft import fft, ifft
from numba import jit, int64


def rk4step(f, y0, dt, args=()):
    """ Takes a single step using RK4 method """
    
    
    k1 = f(y0, *args)
    
   
    
    k2 = f(y0 + 0.5*dt*k1, *args)
    k3 = f(y0 + 0.5*dt*k2, *args)
    k4 = f(y0 + dt*k3, *args)
    
    return y0 + (k1 + 2.*k2 + 2.*k3 + k4)*dt / 6.

#@jit("f8[:](f8[:],u1,f4)")
#@jit (nopython =True)
def calc_density(pos, ncells, L):
    
    
    dx = L/ncells
    pos = pos/dx
    #plower = pos.astype(np.int)
    plower = np.int_(pos)
    
    offset = pos - plower
    opp_offset = 1- offset
    density = zeros((ncells),dtype=np.float64)
    
    
    
    for i in range(0,ncells):
        ind = np.where((plower == i))
        
        density[(i+1)%ncells] += sum(offset[ind])
        density[i] += sum(opp_offset[ind])
        
    density *= float(ncells) / float(len(pos))
    
    
    return (density)
@jit (nopython =True)
#jit("f8[:](f8[:], f8[:])")
def periodic_interp(y, x):

    
    ny = len(y)
    if len(x) > 1:
        y = array(y) # Make sure it's a NumPy array for array indexing
    #xl = floor(x).astype(int) # Left index
    #d=(np.int64(p))
    xl = np.int_(floor(x))
    dx = x - xl
    xl = ((xl % ny) + ny) % ny  # Ensures between 0 and ny-1 inclusive
    
    
    
    return y[xl]*(1. - dx) + y[(xl+1)%ny]*dx


def fft_integrate(y):
    """ Integrate a periodic function using FFTs
    """
    n = len(y) # Get the length of y
    
    f = fft(y) # Take FFT
    # Result is in standard layout with positive frequencies first then negative
    # n even: [ f(0), f(1), ... f(n/2), f(1-n/2) ... f(-1) ]
    # n odd:  [ f(0), f(1), ... f((n-1)/2), f(-(n-1)/2) ... f(-1) ]
    
    if n % 2 == 0: # If an even number of points
        k = concatenate( (arange(0, n/2+1), arange(1-n/2, 0)) )
    else:
        k = concatenate( (arange(0, (n-1)/2+1), arange( -(n-1)/2, 0)) )
    k = 2.*pi*k/n
    
    # Modify frequencies by dividing by ik
    f[1:] /= (1j * k[1:]) 
    f[0] = 0. # Set the arbitrary zero-frequency term to zero
    
    return ifft(f).real # Reverse Fourier Transform
   
def pic(f, ncells, L):
    """ f contains the position and velocity of all particles
    """
    
    nparticles = len(f)//2     # Two values for each particle
    pos = f[0:nparticles] # Position of each particle
    vel = f[nparticles:]      # Velocity of each particle

    dx = L / float(ncells)    # Cell spacing

    # Ensure that pos is between 0 and L
    pos = ((pos % L) + L) % L
    
    # Calculate number density, normalised so 1 when uniform
    density = calc_density(pos, ncells, L)
    
    # Subtract ion density to get total charge density
    rho = density - 1.
    
    # Calculate electric field
    E = -fft_integrate(rho)*dx
    
    # Interpolate E field at particle locations
    accel = -periodic_interp(E, pos/dx)

    # Put back into a single array
    return concatenate( (vel, accel) )

####################################################################
#@jit("f8[:],f8[:](f8[:], f8[:],f8,u1,f8[:], f8[:], f4)")
def run(pos, vel, L, ncells=None, out=[], output_times=linspace(0,20,100), cfl=0.5):
    
    if ncells == None:
        ncells = int(sqrt(len(pos))) # A sensible default

    dx = L / float(ncells)
    
    f = concatenate( (pos, vel) )   # Starting state
    nparticles = len(pos)
    
    time = 0.0
    for tnext in output_times:
        # Advance to tnext
        stepping = True
        dt = cfl * dx / max(abs(vel))
        while stepping:
            # Maximum distance a particle can move is one cell
            
            if time + dt >= tnext:
                # Next time will hit or exceed required output time
                stepping = False
                dt = tnext - time
            #print "Time: ", time, dt
            f = rk4step(pic, f, dt, args=(ncells, L))
            time += dt
            
        # Extract position and velocities
        pos = ((f[0:nparticles] % L) + L) % L
        vel = f[nparticles:]
        
        # Send to output functions
        for func in out:
            func(pos, vel, ncells, L, time)
        
    return pos, vel

####################################################################
# 
# Output functions and classes
#

class Plot:
    """
    Displays three plots: phase space, charge density, and velocity distribution
    """
    def __init__(self, pos, vel, ncells, L):
        
        d = calc_density(pos, ncells, L)
        vhist, bins  = histogram(vel, int(sqrt(len(vel))))
        vbins = 0.5*(bins[1:]+bins[:-1])
        
        # Plot initial positions
        if got_gridspec:
            self.fig = plt.figure()
            self.gs = gridspec.GridSpec(4, 4)
            ax = self.fig.add_subplot(self.gs[0:3,0:3])
            self.phase_plot = ax.plot(pos, vel, '.')[0]
            ax.set_title("Phase space")
            
            ax = self.fig.add_subplot(self.gs[3,0:3])
            self.density_plot = ax.plot(linspace(0, L, ncells), d)[0]
            
            ax = self.fig.add_subplot(self.gs[0:3,3])
            self.vel_plot = ax.plot(vhist, vbins)[0]
        else:
            self.fig = plt.figure()
            self.phase_plot = plt.plot(pos, vel, '.')[0]
            
            self.fig = plt.figure()
            self.density_plot = plt.plot(linspace(0, L, ncells), d)[0]
            
            self.fig = plt.figure()
            self.vel_plot = plt.plot(vhist, vbins)[0]
        plt.ion()
        plt.show()
        
    def __call__(self, pos, vel, ncells, L, t):
        d = calc_density(pos, ncells, L)
        vhist, bins  = histogram(vel, int(sqrt(len(vel))))
        vbins = 0.5*(bins[1:]+bins[:-1])
        
        self.phase_plot.set_data(pos, vel) # Update the plot
        self.density_plot.set_data(linspace(0, L, ncells), d)
        self.vel_plot.set_data(vhist, vbins)
        plt.draw()
        plt.pause(0.05)

class Summary:
    def __init__(self):
        self.t = []
        self.firstharmonic = []
        
    def __call__(self, pos, vel, ncells, L, t):
        # Calculate the charge density
        d = calc_density(pos, ncells, L)
        
        # Amplitude of the first harmonic
        fh = 2.*abs(fft(d)[1]) / float(ncells)
        
        #print ("Time:", t, "First:", fh)
        
        self.t.append(t)
        self.firstharmonic.append(fh)

####################################################################
# 
# Functions to create the initial conditions
#

def landau(npart, L, alpha=0.2):
    """
    Creates the initial conditions for Landau damping
    
    """
    # Start with a uniform distribution of positions
    pos = random.uniform(0., L, npart)
    pos0 = pos.copy()
    k = 2.*pi / L
    for i in range(10): # Adjust distribution using Newton iterations
        pos -= ( pos + alpha*sin(k*pos)/k - pos0 ) / ( 1. + alpha*cos(k*pos) )
        
    # Normal velocity distribution
    vel = random.normal(0.0, 1.0, npart)
    
    return pos, vel

def twostream(npart, L, vbeam=2):
    # Start with a uniform distribution of positions
    pos = random.uniform(0., L, npart)
    # Normal velocity distribution
    vel = random.normal(0.0, 1.0, npart)
    
    np2 = int(npart / 2)
    vel[:np2] += vbeam  # Half the particles moving one way
    vel[np2:] -= vbeam  # and half the other
    
    return pos,vel

####################################################################

def main():
    average_time = []
    for i in range(10):
        start_time = time.perf_counter()
        L = 4.*pi
        pos, vel = landau(10000, L)
        
      
        s = Summary()
        
        run(pos, vel, L, 10, [s], linspace(0.,30,75))
        
        end_time = time.perf_counter()
        
        time_difference = end_time - start_time
        
        
        
        average_time.append(time_difference)
    
    
    print ("Average time is", np.round(np.mean(average_time),2), "+/-", np.round(np.std(average_time),2))

main()

    
