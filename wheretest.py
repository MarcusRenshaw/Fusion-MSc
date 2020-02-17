# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 23:11:13 2018

@author: Knowhow
"""
import numpy as np



    






"""

ind = np.where(pos == 0)

print (ind)

density = zeros([ncells])

nparticles = 5.

dx = L / ncells       # Uniform cell spacing


pos = np.array([4.352,6.754,9.625,1.234, 0.182])

for p in pos/ dx:    # Loop over all the particles, converting position into a cell number
    
    
    plower = int(p)        # Cell to the left (rounding down)
    
    
    offset = p - plower    # Offset from the left
    
    density[plower] += 1. - offset

    density[(plower + 1) % ncells] += offset
    
    ind = np.where(plower == 7)
    print (ind)
"""        

