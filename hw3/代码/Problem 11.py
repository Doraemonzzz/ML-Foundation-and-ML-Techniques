# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:05:13 2019

@author: qinzhen
"""

import numpy as np

X = np.array(
        [[1, 1, 1, 1, 1, 1],
         [1, 1, -1, 1, -1, 1],
         [1, -1, -1, 1, 1, 1],
         [1, -1, 1, 1, -1, 1],
         [1, 0, 0, 0, 0, 0],
         [1, 1, 0, 1, 0, 0]]
        )
print(np.linalg.det(X))
