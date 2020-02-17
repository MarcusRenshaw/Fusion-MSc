# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:24:27 2017

@author: Knowhow
"""



import matplotlib.pyplot as plt


          
def test_func(x, a, b):

    return a * np.sin(b * x)

from scipy import optimize

params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=[2, 2])

print(params)

print (a,b)