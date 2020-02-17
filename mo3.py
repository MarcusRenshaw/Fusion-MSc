# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 11:58:43 2018

@author: Knowhow
"""

# -*- coding: utf-8 -*-
"""
Landau Damping 1D Model Analysis - Noise Analysis

Author Mohamad Abdallah

@author: ma1439
"""
from   mo2 import *
from   scipy.signal import argrelextrema

import os
import numpy as np

Noise            = []
MaxPeak          = []
MinPeak          = []
Loaded_Raw_Data  = []
Loaded_Time_Data = []
ArrayofPeaks     = []
ArrayofTimes     = [] 
    
for Y in [10, 50, 100, 150, 200]: #Looping over cell size
    for Z in [100, 1000, 25000, 50000, 100000]: 
    #Looping over number of particles per cell size
        for X in range(10): #Loop for all data sets per cell size, per particle size
            Loaded_Raw_Data  = np.loadtxt('Raw_Data/Cell{i}/Raw_data_{j}/Particles_{j}/Raw_data_{k}'.format(i=Y, j=Z, k=X))
            Loaded_Time_Data = np.loadtxt('Raw_Data/Cell{i}/Raw_data_{j}/Particles_{j}/Raw_time_data_{k}'.format(i=Y, j=Z, k=X))
                
            #The above code reads time data and first harmonic data into two new arrays

            #called Loaded_Time_Data and Loaded_Raw_Data, these arrays contain the data for

            #each individual run of define loop from x in defined range  
       
            Peaks = argrelextrema(np.array(Loaded_Raw_Data), np.greater)[0]
    
            #The above function argrelextrema finds the highest peak point in the array
    
            #defined, which in this case is Loaded_Raw_Data. The returned values are defined 

            #by the Peaks variable
            

            for i in Peaks:
                ArrayofPeaks.append(Loaded_Raw_Data[i])
                ArrayofTimes.append(Loaded_Time_Data[i])
        
            Stat = False
            for i in range(len(ArrayofPeaks)-1):
                if ArrayofPeaks[i+1] > ArrayofPeaks[i]:
                    MaxPeak.append(ArrayofPeaks[i+1:])
                    Stat = True
                    MinPeak.append(ArrayofPeaks[i])
                    break
                if not Stat: MaxPeak.append(ArrayofPeaks[-1])
            
                Noise.append(np.mean(MaxPeak))

        AvgNoise = np.mean(Noise)
        
print (AvgNoise)