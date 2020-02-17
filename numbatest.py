

from numpy import array
from numba import jit

@jit
def test():
    a = array([0,0])
    return a

blah = test()

