from pictry import *
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

length=[]
angular_frequency = []
speed =[]

for y in [6,8,10,12,14,16,18,24,48]: 
    peakdifference = []
    
    newpath = r"C:\Users\Knowhow\Documents\Python Scripts\Length{length}".format(length = y)     
    directory = os.listdir(newpath) # dir is your directory path
    number_files = len(directory)
    
    if number_files > 1:
        for x in range(1,number_files+1):
            
            filename = r"C:\Users\Knowhow\Documents\Python Scripts\Length{length}\{runnum}".format(length=y,runnum=x)
            loadeddata = np.loadtxt(filename)
            
            data = loadeddata[1,:]
            timedata = loadeddata[0,:]
            #timedata = loadeddata[0:(len(loadeddata)//2)]
            #data = loadeddata[len(loadeddata)//2:]
            
            #firstpeakindex = [0]
            index = argrelextrema(np.array(data), np.greater)[0]
            #firstpeakindex.extend(index)
            time = []   #timeofpeaks
            peaks = []
            
            for i in index:
                peaks.append(data[i])
                time.append(timedata[i])
            
            low=[]
            a=False
            higherpeaks = []
            lowtime = []
        
            for i in range(len(peaks)-1):
                if peaks[i+1] > peaks[i]:
                    higherpeaks.append(peaks[i+1:])
                    a=True
                    low.append(peaks[i])
                    lowtime.append(time[i])
                    noiseindex = i
                    break
            if not a: higherpeaks.append(peaks[-1])
            
            
            for i in range(len(time[:noiseindex+1])):
                peakdifference.append(np.pi/(time[i+1] - time[i]))
                
            omega = np.pi/(time[i+1] - time[i])
            
        averagedifference = np.mean(peakdifference)
        
        omega = np.mean(peakdifference)
        wavenumber = 2*np.pi / y
        
        velocity = omega/wavenumber
        
        speed.append(velocity)
        
        length.append(y)
        angular_frequency.append(averagedifference)
        
       
    else:
        break

print (length)   
#print (angular_frequency)
print (speed)


plt.plot(length, speed)
plt.xlabel('Length of Box')
plt.ylabel('Speed')
plt.title('Angular frequency for varying Length of Box')
  
plt.show()
    


        

