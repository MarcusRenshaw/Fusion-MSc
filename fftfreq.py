# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:41:29 2017

@author: Knowhow
"""

from scipy.fftpack import fftfreq
freq = fftfreq(8, 0.125)
print (freq)