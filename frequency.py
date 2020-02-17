from pictry import *
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


cells = []
particles = []
ang_freq = []

#for z in [25,50,75,100,300]:#range(2,50,2):      #no. of cells
#for y in [1000, 5000, 10000, 20000, 40000, 60000,80000]:#100000,125000,150000,200000]: 
for z in [20]:#range(25,50,75,100,300):      #no. of cells
    for y in [200000]:           #no. of cells
    
        peakdifference=[]
        newpath = r"C:\Users\Knowhow\Documents\Python Scripts\NumCells{cells}\NumPart{parts}".format(cells=z,parts=y)        
        directory = os.listdir(newpath) # dir is your directory path
        number_files = len(directory)
        if number_files > 1:
            for x in range(1,number_files+1):
                
                filename = r"C:\Users\Knowhow\Documents\Python Scripts\NumCells{cells}\NumPart{parts}\{runnum}".format(cells=z,parts=y,runnum=x)
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
                    
                    
                    averagedifference = np.mean(peakdifference)

                
            cells.append(z)
            particles.append(y)
            ang_freq.append(averagedifference)
            print (ang_freq)        
        else:
            break
        


print ("The angular frequency is:", ang_freq, "+-",error)

ax.scatter(cells, particles,ang_freq,c='r',marker='o')

ax.set_xlabel('No. of cells')
ax.set_ylabel('No. of particles')
ax.set_zlabel('Angular frequency s^-1')
ax.set_title("Angular frequency with varying number of cells and particles")


plt.show()
        
    
     
            
    

        

