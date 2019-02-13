# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 00:51:11 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#读取数据
data = np.genfromtxt("data.txt")
#获取维度
n, d = data.shape
#分离X
X = data[:, :-1]
#添加偏置项1
X = np.c_[np.ones(n), X]
#分离y
y = data[:, -1]


#problem 15    
print(hlp.PLA(X, y))

#problem 16
hlp.f1(hlp.PLA, X, y, 2000, 1)

#problem 17
hlp.f1(hlp.PLA, X, y, 2000, 0.5)