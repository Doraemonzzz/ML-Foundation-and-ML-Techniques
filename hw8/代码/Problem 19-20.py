# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 12:03:28 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
train = np.loadtxt("hw8_nolabel_train.dat")

def K_mean(data, k):
    data = train
    N = data.shape[0]
    #初始化中心
    index = np.random.choice(range(N), size=k, replace=False )
    center = data[index]
    newcenter = np.copy(center)
    label = np.zeros(N)
    while True:
        #Step 1:计算距离最近的中心
        for i in range(N):
            distance = np.sum((data[i] - center)**2, axis = 1)
            label[i] = np.argmin(distance)
        #Step 2:计算新的中心
        for i in range(k):
            newcenter[i] = np.mean(data[label == i])
        #判断是否收敛
        if np.sum((newcenter - center)**2) < 1e-5:
            break
        center = np.copy(newcenter)
    
    #计算误差
    Error = 0
    for i in range(k):
        Error += np.sum((data[label == i] - center[i])**2)
    return Error/N

K = [2, 4, 6, 8, 10]
Ein = []
for k in K:
    Ein.append(K_mean(train, k))
    
plt.scatter(K, Ein)
plt.title("$K$ VS $E_{in}$")
plt.xlabel("$K$")
plt.ylabel("$E_{in}$")
plt.show()

