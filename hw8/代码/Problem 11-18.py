# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:38:49 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
train = np.loadtxt("hw8_train.dat")
test = np.loadtxt("hw8_test.dat")

def Knn(x, data, k):
    distance = np.sum((x[:-1] - data[:, :-1])**2, axis = 1)
    index = np.argsort(distance)[:k]
    label = data[:, -1][index]
    return np.sign(np.sum(label))

def Error(data1, data2 ,k):
    error = 0
    for x in data1:
        #判断label是否相等
        error += Knn(x, data2, k) != x[-1]
    return error/data1.shape[0]

# Q11-12
K = [1, 3, 5, 7, 9]
Ein = []
for k in K:
    Ein.append(Error(train, train, k))

plt.scatter(K, Ein)
plt.title("$K$ VS $E_{in}$")
plt.xlabel("$K$")
plt.ylabel("$E_{in}$")
plt.show()

# Q13-14
K = [1, 3, 5, 7, 9]
Eout = []
for k in K:
    Eout.append(Error(test, train, k))

plt.scatter(K, Eout)
plt.title("$K$ VS $E_{out}$")
plt.xlabel("$K$")
plt.ylabel("$E_{out}$")
plt.show()


def Knn_new(x, data, gamma):
    distance = np.sum((x[:-1] - data[:, :-1])**2, axis = 1)
    Exp = np.exp(-gamma * distance)
    S = data[:, -1] * Exp
    return np.sign(np.sum(S))

def Error_new(data1, data2, gamma):
    error = 0
    for x in data1:
        #判断label是否相等
        error += Knn_new(x, data2, gamma) != x[-1]
    return error/data1.shape[0]

# Q15-16
Gamma = [0.001, 0.1, 1, 10, 100]
Ein_new = []
for gamma in Gamma:
    Ein_new.append(Error_new(train, train, gamma))

plt.scatter(Gamma, Ein_new)
plt.title("$\gamma$ VS $E_{in}$")
plt.xlabel("$\gamma$")
plt.ylabel("$E_{in}$")
plt.show()

# Q17-18
Gamma = [0.001, 0.1, 1, 10, 100]
Eout_new = []
for gamma in Gamma:
    Eout_new.append(Error_new(test, train, gamma))

plt.scatter(Gamma, Eout_new)
plt.title("$\gamma$ VS $E_{out}$")
plt.xlabel("$\gamma$")
plt.ylabel("$E_{out}$")
plt.show()

'''
data = train
k = 3
x = train[0]
distance = np.sum((x[:-1] - data[:, :-1])**2, axis = 1)
index = np.argsort(distance)[:k]
label = data[:, -1][index]
np.sign(np.sum(label))
'''