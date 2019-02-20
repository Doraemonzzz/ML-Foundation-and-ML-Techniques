# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 20:12:42 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

dvc = 50
delta = 0.05

#计算ln(m(N))
def lm(n):
    return dvc * np.log(n)

#Original VC-bound
def f1(n):
    result = (8 / n) * (np.log(4 / delta) + lm(2 * n))
    result = result ** 0.5
    return result

#Variant VC bound
def f2(n):
    result = (16 / n) * (np.log(2 / (delta ** 0.5)) + lm(n))
    result = result ** 0.5
    return result

#Rademacher Penalty Bound
def f3(n):
    k1 = 2 * (np.log(2 * n) + lm(n)) / n
    k2 = (2 / n) * np.log(1 / delta)
    k3 = 1 / n
    result = k1 ** 0.5 + k2 ** 0.5 + k3
    return result

#Parrondo and Van den Broek
def f4(n):
    k1 = 1 / n
    k2 = 1 / (n ** 2) + (1 / n) * (np.log(6 / delta) + lm(2 * n))
    k2 = k2 ** 0.5
    result = k1 + k2
    return result

#Devroye
def f5(n):
    k1 = 1 / (n - 2)
    k2 = (np.log(4 / delta) + lm(n * n)) / (2 * (n - 2)) + 1 / ((n - 2) ** 2)
    k2 = k2 ** 0.5
    result = k1 + k2
    return result

#### Problem 4
#产生点集
x = np.arange(100, 2000)

y1 = [f1(i) for i in x]
y2 = [f2(i) for i in x]
y3 = [f3(i) for i in x]
y4 = [f4(i) for i in x]
y5 = [f5(i) for i in x]

plt.plot(x, y1, label="Original VC-bound")
plt.plot(x, y2, label="Variant VC-bound")
plt.plot(x, y3, label="Rademacher Penalty Bound")
plt.plot(x, y4, label="Parrondo and Van den Broek")
plt.plot(x, y5, label="Devroye")
plt.legend()
plt.show()

#比较y4, y5
plt.plot(x, y4, label="Parrondo and Van den Broek")
plt.plot(x, y5, label="Devroye")
plt.legend()
plt.show()

#### Problem 5
x = np.arange(3, 11)
y1 = [f1(i) for i in x]
y2 = [f2(i) for i in x]
y3 = [f3(i) for i in x]
y4 = [f4(i) for i in x]
y5 = [f5(i) for i in x]

plt.plot(x, y1, label="Original VC-bound")
plt.plot(x, y2, label="Variant VC-bound")
plt.plot(x, y3, label="Rademacher Penalty Bound")
plt.plot(x, y4, label="Parrondo and Van den Broek")
plt.plot(x, y5, label="Devroye")
plt.legend()
plt.show()
