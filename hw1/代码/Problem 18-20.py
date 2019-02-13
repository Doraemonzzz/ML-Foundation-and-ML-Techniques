# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 01:14:27 2019

@author: qinzhen
"""

import matplotlib.pyplot as plt
import numpy as np
import helper as hlp
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

data_train = np.genfromtxt("hw1_18_train.txt")
data_test = np.genfromtxt("hw1_18_test.txt")

X_train, y_train = hlp.preprocess(data_train)
X_test, y_test = hlp.preprocess(data_test)

#problem 18
hlp.f2(hlp.Pocket_PLA, X_train, y_train, X_test, y_test, 2000, max_step=50)

#problem 19
result = hlp.f2(hlp.PLA, X_train, y_train, X_test, y_test, 2000, 1, max_step=50)

#problem 20
hlp.f2(hlp.Pocket_PLA, X_train, y_train, X_test, y_test, 2000, max_step=100)