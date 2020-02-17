# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:49:31 2017

@author: Knowhow
"""

def getForce(x,y):
    #Create array for force values
    N=len(x) ; Fx = zeros(N) ; Fy = zeros(N)
    #Now calculate the force on each particle
    for i in range(N):
        #Loop through all the other particles
        #to calculate their force on this particle
        for j in range(N):
            if i==j: continue #No force on self
            #Calculate distance
            dx = x[i] - x[j] ; dy = y[i] - y[j]
            R=sqrt(dx**2+dy**2)
            #Calculate force
            Fx[i] -= dx/R**3 ; Fy[i] -= dy/R**3
    return Fx, Fy
    
def f(state, time):
    #Work out N from state
    N = int(len(state) / 4)

    #Unpack variables
    x = state[0:N] ; y = state[N:(2*N)]
    vx = state[(2*N):3*N] ; vy = state[(3*N):]

    #Find out the force
    Fx, Fy = getForce(x,y)

    #Concatenate takes list of B, N length vectors
    #and returns a single B*N vector
    #Note dx/dt=vx, dy/dt=vy, dvx/dt=Fx and dvy/dt=Fy
    return concatenate([ vx, vy, Fx, Fy ])


from numpy import zeros, sqrt, concatenate, linspace
from scipy.integrate import odeint
#Initial conditions: 
#Pcle 1 at (x,y)=(-1,0), Pcle 2 at(x,y)=(1,)
#Pcle 1 moving with (vx,vy)=(0,0.2), Pcle 2 with (vx,vy)=(0,-0.2)
initial = [-1.0, 1.0, 0.0,0.0, 0.0, 0.0, 0.2,-0.2] 
#Times of interest
t = linspace(0, 20, 200)
#Integrate
result = odeint(f, initial, t)
#Plot
import matplotlib.pyplot as plt
plt.plot( result[:,0], result[:,2])
plt.plot( result[:,1], result[:,3])
plt.show()