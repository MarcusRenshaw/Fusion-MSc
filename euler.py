
from numpy import linspace, pi, array, ndarray, vstack

def eulerstep(f, y0, dt):
    """ Takes a single step using the Euler method """
    return y0 + dt * f(y0)

def solve(f, y0, time):
    """ Solve a system of equations dy/dt = f(y) using Euler's method"""
    y = array(y0)  # Starting value
    n = len(time) # Number of time points 
    
    result = ndarray(shape=(n, len(y)))  # Allocate the array for the result
    result[0,:] = y  # Insert the first point

    for i in range(1, len(time)):
        y = eulerstep(f, y, time[i] - time[i-1])
        result[i,:] = y
    return result

if __name__ == "__main__":
    # Test case of Simple Harmonic Motion
    
    def sho(y):
        return array([ y[1], -y[0] ])
    
    y0 = array([1,0]) # Starting values
    nt = 40
    t = linspace(0, 4*pi, nt) # Times for the output
    y = solve(sho, y0, t)
        
    import matplotlib.pyplot as plt
    from numpy import cos
    
    plt.plot(t / (2.*pi), y[:, 0], label='Euler, %d steps' % nt)
    plt.plot(t / (2.*pi), cos(t), label='cos(t)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlabel("Periods")
    plt.ylabel("y")
    plt.title("Euler's method")
    plt.show()
