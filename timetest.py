# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:20:44 2018

@author: Knowhow
"""


from pictry import *
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
import time

#for z in [25,50,75,100,300]:#range(2,50,2):      #no. of cells
#for y in [1000, 5000, 10000, 20000, 40000, 60000,80000]:#100000,125000,150000,200000]: 
    
for z in [300]:#range(2,50,2):      #no. of cells
    for y in [40000, 60000,80000]:#100000,125000,150000,200000]: 
        newpath = r"C:\Users\Knowhow\Documents\Python Scripts\TimeTest\NumCells{cells}\NumPart{parts}".format(cells=z,parts=y)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        
        
        for x in range(1,4):       #produces files (1,11) produces 10 files
            
            start_time = time.time()
            L = 4*np.pi
            pos, vel = landau(y, L)
            s = Summary()
            run(pos, vel, L, z, [s], linspace(0.,35,90))
            
            
            index = argrelextrema(np.array(s.firstharmonic), np.greater)[0]
            peaktime = []
            peaks = []
            for i in index:
                peaks.append(s.firstharmonic[i])
                peaktime.append(s.t[i])
            """
            plt.plot(s.t,s.firstharmonic)
            plt.yscale('log')
            
            plt.plot(time, peaks,'x')
            plt.title("Amplitude in time")
            plt.xlabel("Time, s")
            plt.ylabel("Amplitude, [normalised]")
            plt.show()
            """
            
            
            
            end_time = time.time()
    
            time_difference = end_time - start_time
            
            
            
        
            filename = r"C:\Users\Knowhow\Documents\Python Scripts\TimeTest\NumCells{cells}\NumPart{parts}\{runnum}".format(cells=z,parts=y,runnum=x)
            np.savetxt(filename, (x,time_difference))
            
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
            
            #np.savetxt('data.txt_{i}'.format(i=x),(s.t, s.firstharmonic), newline = ' ')
    
    #filename = r"C:\Users\Knowhow\Documents\Python Scripts\Lengths{lval}\{runnum}".format(lval=y,runnum=x)
    #np.savetxt(filename, (s.t, s.firstharmonic))
    
        

