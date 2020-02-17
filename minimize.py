# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 20:48:11 2017

@author: Knowhow
"""
from scipy import optimize
import numpy as np

def f(x):

    return x**2 + 10*np.sin(x)

result = optimize.minimize(f, x0=-5)


print (result)
print (result.x)
print (result.fun)

print (f(-1.30))

other = optimize.basinhopping(f, 0)

print (other.x)
