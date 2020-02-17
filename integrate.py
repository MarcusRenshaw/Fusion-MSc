from scipy.integrate import odeint
from numpy import linspace, exp
import matplotlib.pyplot as plt

#Here we define our function which returns df/dt = -a*f
#Note we have assumed that a = 10

def dfdt(curF, curT, a=10):
    #we dont have to do anything with curT
    return -a*curF

#Now define the times at which we want to know the result
time=linspace(0,1,40)
    
#Set the initial condition
f0=10

#WHich a values do we want to use?
avals=[0,1,10,30,-1]

for a in avals:
    #Now we can solve the equation for this a
    #Note we need a comma after the a here to
    #make sure args is a tuple
    result = odeint(dfdt,f0,time,args=(a,))
    
    plt.plot(time,result,label='a = '+str(a))
    
plt.xlabel('Time'); plt.ylabel('f')
plt.legend(loc='best') ; plt.show()