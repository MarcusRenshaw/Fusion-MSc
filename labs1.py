
"""import numpy as np
myarray = np.array([0,2,4,6,8,10])

b = myarray.shape
"""
from scipy.optimize import curve_fit
import numpy as np
import icf

def gaussian(x,*params):
    
    A = params[0]
    x0 = params[1]
    c = params[2]
    return A*np.exp(-(x-x0)**2/(2*c*c))

xdata = np.linspace(0, 4, 100)
ydata = gaussian(xdata, 3, 2, 0.2)

for i in range(len(ydata)):
    ydata[i] += 0.4*(np.random.random_sample()-0.5)
    
guess = [1,1,1]
print("Our initial guess is", guess)
popt,pcov = curve_fit(gaussian, xdata, ydata, p0=guess)
    
for i in range(len(popt)):
    print ("Paramter", i,":", popt[i],"+-", np.sqrt(pcov[i][i]))
    
yfit = gaussian(xdata, *popt)

print("R^2 = ", icf.r_squared(ydata, yfit))  

icf.fit_plot(xdata, ydata, yfit)


