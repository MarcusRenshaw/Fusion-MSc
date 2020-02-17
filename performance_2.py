# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:11:36 2018

@author: Knowhow
"""

from Speedtest import *
import matplotlib.pyplot as plt
import numpy as np

import time


average_time = []
for i in range(1):
    
    
    
    start_time = time.perf_counter()
    L = 4.*pi
    pos, vel = landau(10000, L)
    
  
    s = Summary()
    
    run(pos, vel, L, 10, [s], linspace(0.,30,75))
    
    end_time = time.perf_counter()
    
    time_difference = end_time - start_time
    
    
    
    average_time.append(time_difference)


print ("Average time is", np.round(np.mean(average_time),2), "+/-", np.round(np.std(average_time),2))




r"""
#SET UP SEED VALUES
filename = r"C:\Users\Knowhow\Documents\Python Scripts\seed_data_pos.txt"
#np.savetxt(filename, pos)
filename2=r"C:\Users\Knowhow\Documents\Python Scripts\seed_data_vel.txt"
#vel = np.savetxt(filename2, vel)

#LOAD SEED ARRAYS
pos = np.loadtxt(filename)
vel = np.loadtxt(filename2)
"""




