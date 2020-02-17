# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:49:12 2017

@author: Knowhow
"""

import numpy as np
from scipy import linalg

arr = np.array([[1, 2],

                [3, 4]])

print (linalg.det(arr))





arr = np.array([[1, 2],

                [3, 4]])

iarr = linalg.inv(arr)

print (iarr)