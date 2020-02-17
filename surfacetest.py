
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
time = []

for z in [25,50,75,100,300]:#range(2,50,2):      #no. of cells
    for y in [1000,5000,10000,20000, 40000, 60000,80000]: 
        totaltimevalues = []
        newpath = r"C:\Users\Knowhow\Documents\Python Scripts\TimeTest\NumCells{cells}\NumPart{parts}".format(cells=z,parts=y)        
        directory = os.listdir(newpath) # dir is your directory path
        number_files = len(directory)
        
        if number_files > 1:
            for x in range(1,number_files+1):
                
                filename = r"C:\Users\Knowhow\Documents\Python Scripts\TimeTest\NumCells{cells}\NumPart{parts}\{runnum}".format(cells=z,parts=y,runnum=x)
                loadeddata = np.loadtxt(filename)
            
                #data = loadeddata[1,:]
                time_value = loadeddata[1]
                
                
                
                totaltimevalues.append(time_value)
                
            
            
            average_time = np.mean(totaltimevalues)
            
            
                
 
            
            cells.append(z)
            particles.append(y)
            time.append(average_time)
        else:
            break
        
print (time)

     
ax.scatter(cells, particles,time,c='r',marker='o')

ax.set_xlabel('no. of cells')
ax.set_ylabel('no. of particles')
ax.set_zlabel('noise amplitude')
ax.set_title("Noise amplitude for vary no. of cells and no. of particles")
#ax.set_ylim3d(0,100000)

plt.show()


"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sys import argv


df = pd.DataFrame({'x': cells, 'y': particles, 'z': time})

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('teste.pdf')
plt.show()
"""

    


        

