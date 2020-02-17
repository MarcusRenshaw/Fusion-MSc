# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:25:33 2017

@author: Knowhow
"""
import numpy as np
from scipy import io as spio

a = np.ones((3, 3))

spio.savemat('file.mat', {'a': a}) # savemat expects a dictionary

data = spio.loadmat('file.mat')

data['a']