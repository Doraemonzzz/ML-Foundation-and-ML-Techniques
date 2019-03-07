# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:22:27 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#构造损失函数
def e1(s):
    if s > 0:
        return 0
    else:
        return 1

def e2(s):
    return max(0, 1 - s)

def e3(s):
    t = max(0, 1 - s)
    return t ** 2

def e4(s):
    return max(0, -s)

def e5(s):
    return 1 / (1 + np.exp(s))

def e6(s):
    return np.exp(-s)

x = np.arange(-1, 1, 0.01)

y1 = [e1(i) for i in x]
y2 = [e2(i) for i in x]
y3 = [e3(i) for i in x]
y4 = [e4(i) for i in x]
y5 = [e5(i) for i in x]
y6 = [e6(i) for i in x]

plt.plot(x, y1, label='e1')
plt.plot(x, y2, label='e2')
plt.plot(x, y3, label='e3')
plt.plot(x, y4, label='e4')
plt.plot(x, y5, label='e5')
plt.plot(x ,y6, label='e6')
plt.legend()
plt.title('损失函数比较')
plt.show()