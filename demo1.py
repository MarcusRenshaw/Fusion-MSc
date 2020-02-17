# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 14:11:16 2017

@author: Knowhow
"""

# Lotka-Volterra predator-prey
#
# x = prey 
# y = predator

def lv(state, time, alpha, beta, gamma, delta):
    # inputs are system state and
    # the simulation time
    
    x = state[0]
    y = state[1]
    dxdt = alpha * x - beta*x*y
    dydt = -gamma*y + delta*x*y

    
    return [dxdt, dydt]

from scipy.integrate import odeint
from numpy import linspace
initial = [5, 5]
t = linspace(0.0, 20, 500)
result = odeint(lv, initial, t, args=(1.0, 1.0, 3.0, 1.0))

import matplotlib.pyplot as plt

plt.plot(t, result[:, 0], label="prey")
plt.plot(t, result[:,1], label="predator")
plt.legend()
plt.show()



