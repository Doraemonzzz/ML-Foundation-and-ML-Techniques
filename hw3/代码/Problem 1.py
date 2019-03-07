# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:15:13 2019

@author: qinzhen
"""

def f(d, delta, Ein):
    return (d + 1) / (1 - Ein / (delta ** 2))

print(f(8, 0.1, 0.008))