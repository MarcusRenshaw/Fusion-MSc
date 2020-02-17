from pictry import *
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cells = []
particles = []
noise = []

for z in [50]:#range(25,50,75,100,300):      #no. of cells
    totalerrorvalues =[]
    for y in [1000, 5000, 10000, 20000, 40000, 60000,80000]: 
        totalnoisevalues = []
        
        
        newpath = r"C:\Users\Knowhow\Documents\Python Scripts\NumCells{cells}\NumPart{parts}".format(cells=z,parts=y)        
        directory = os.listdir(newpath) # dir is your directory path
        number_files = len(directory)
        
        if number_files > 1:
            for x in range(1,number_files+1):
                
                filename = r"C:\Users\Knowhow\Documents\Python Scripts\NumCells{cells}\NumPart{parts}\{runnum}".format(cells=z,parts=y,runnum=x)
                loadeddata = np.loadtxt(filename)
            
                
                data = loadeddata[1,:]
                timedata = loadeddata[0,:]
                
                index = argrelextrema(np.array(data), np.greater)[0]
                time = []
                peaks = []
            
                peaks.append(data[0])
                time.append(timedata[0])
                for i in index:
                    peaks.append(data[i])
                    time.append(timedata[i])
            
                low=[]
                a=False
                higherpeaks = []
                for i in range(len(peaks)-1):
                    if peaks[i+1] > peaks[i]:
                        higherpeaks.append(peaks[i+1:])
                        a=True
                        low.append(peaks[i])
                        break
                if not a: higherpeaks.append(peaks[-1])
             
                totalnoisevalues.append(np.mean(higherpeaks))
            
            totalerrorvalues.append(np.std(higherpeaks))
            totalnoiseaverage = np.mean(totalnoisevalues)
            #totalerroraverage = np.mean(totalerrorvalues)
            
            
            cells.append(z)
            particles.append(y)
            noise.append(totalnoiseaverage)
        else:
            break
print (totalerrorvalues)
"""
yerr = totalerrorvalues
plt.figure()
plt.errorbar(particles,noise, yerr)
plt.title("Noise against number of particles with errors")
plt.xlabel("Number of particles")
plt.ylabel("Noise Value")
plt.plot(particles, noise)
plt.show()
"""
"""
new_x = particles
new_y = noise
m , c = np.polyfit(new_x, np.log(new_y),1)

print (m)

from scipy.optimize import curve_fit

def f(x, a, b, c):
    
    
    return a + b * (1-np.exp(-c/x))
    
X = particles
Y = noise
sol= curve_fit(f, X, Y, p0 = (.75,.75,200,) )

plt.plot(X, Y, 'o', label = 'data')
plt.title("Curve fit of noise against number of particles")
plt.xlabel("Number of particles")
plt.ylabel("Noise Value")
plt.plot(X, [f(x,*sol[0]) for x in X], lw = 3, label = 'fit')

print ("The equation of this line is: a + b * (1-np.exp(-c/x))" )
print ("where a=0.0063, b=0.048, c=2420")

#plt.plot(particles, noise)
"""           

   
ax.scatter(cells, particles,noise,c='r',marker='o')

ax.set_xlabel('No. of cells')
ax.set_ylabel('No. of particles')
ax.set_zlabel('Noise amplitude')
ax.set_title("Noise amplitude for vary no. of cells and no. of particles")
ax.set_ylim3d(0,100000)

plt.show()

"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sys import argv


df = pd.DataFrame({'x': cells , 'y': particles, 'z': noise})

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('teste.pdf')
ax.set_xlabel('No. of cells')
ax.set_ylabel('No. of particles')
ax.set_zlabel('Noise amplitude')
ax.set_title("Noise amplitude for vary no. of cells and no. of particles")
plt.show()

"""
    


        

