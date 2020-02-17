# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:34:25 2018

@author: Knowhow
"""

from pictry import *
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os

#newpath = r'C:\Program Files\arbitrary' 
#if not os.path.exists(newpath):
    #os.makedirs(newpath)
    
    

    

for y in [4,6,8,10,12,14,16,18,24,48]:      #no. of particles
    newpath = r"C:\Users\Knowhow\Documents\Python Scripts\Length{length}".format(length=y)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
       