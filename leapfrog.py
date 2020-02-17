# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:42:48 2017

@author: Knowhow
"""

#
#
# 

from numpy import linspace, pi, array, ndarray, vstack
from euler import eulerstep

def leapfrogstep(f, y0, y1, dt):
    """ Takes a single step using the Leapfrog method 
    Needs two previous time points y0 and y1
    """
    return y0 + 2.*dt * f(y1)

def solve(f, y0, time):
    """ Solve a system of equations dy/dt = f(y) using Euler's method"""
    y = y0  # Starting value
    n = len(time) # Number of time points 
    
    result = ndarray(shape=(n, len(y0)))  # Allocate the array for the result
    result[0,:] = y0  # Insert the first point

    # Use Euler's method for the first step
    y1 = eulerstep(f, y, time[1] - time[0])
    result[1,:] = y1

    for i in range(2, len(time)):
        # y1 is the most recent value, and y is the value before
        y2 = leapfrogstep(f, y, y1, time[i] - time[i-1])
        result[i,:] = y2
        # Shift y and y1 along one time point for the next step
        y = y1
        y1 = y2

    return result

if __name__ == "__main__":
    # Test case of Simple Harmonic Motion
    
    def sho(y):
        return array([ y[1], -y[0] ])
    
    y0 = array([1,0]) # Starting values
    nt = 40
    t = linspace(0, 4*pi, nt) # Times for the output
    y = solve(sho, y0, t)
        
    import matplotlib.pyplot as plt
    from numpy import cos
    
    plt.plot(t / (2.*pi), y[:, 0], label='Leapfrog, %d steps' % nt)
    plt.plot(t / (2.*pi), cos(t), label='cos(t)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlabel("Periods")
    plt.ylabel("y")
    plt.title("Leapfrog method")
    plt.show()
