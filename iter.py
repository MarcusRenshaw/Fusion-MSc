# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:46:52 2017

@author: Knowhow
"""

nums = [1, 2, 3]      # note that ... varies: these are different objects




it = iter(nums)

print(next(it))
print(next(it))
print(next(it))

print(list(i for i in nums))




