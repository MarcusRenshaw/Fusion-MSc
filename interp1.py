# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:39:39 2017

@author: Knowhow
"""


import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

x = np.arange(0, 10)
y = np.exp(-x/3.0)
f = interpolate.interp1d(x, y)








