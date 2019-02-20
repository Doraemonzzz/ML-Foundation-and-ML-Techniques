# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 22:51:02 2019

@author: qinzhen
"""

import numpy as np

def decision_stump(X, y):
    #排序
    X1 = np.sort(X)
    #计算theta
    theta = Theta(X1)
    #向量化执行计算
    n = theta.shape[0]
    #将X复制按横轴n份
    X = np.tile(X, (n, 1))
    #s=1
    y1 = np.sign(X - theta)
    #s=-1
    y2 = np.sign(X - theta) * (-1)
    #统计错误
    error1 = np.sum(y1!=y, axis = 1)
    error2 = np.sum(y2!=y, axis = 1)
    #计算最小错误对应的下标
    i1 = np.argmin(error1)
    i2 = np.argmin(error2)
    #判断哪个误差更小
    if error1[i1] < error2[i2]:
        s = 1
        t = theta[i1][0]
        error = error1[i1] / n
    else:
        s = -1
        t = theta[i2][0]
        error = error2[i2] / n
    return s, t, error

def multi_decision_stump(X, y):
    """
    对每个维度使用decision_stump
    """
    n, m = X.shape
    #初始化s, theta, d，最小错误为error
    s = 1
    t = 0
    d = 0
    error = 1
    for i in range(m):
        X1 = X[:, i]
        s0, t0, error0 = decision_stump(X1, y)
        if error0 < error:
            error = error0
            d = i
            t = t0
            s = s0
    return s, t, d, error

def preprocess(data):
    X = data[:, :-1]
    y = data[:, -1]
    
    return X, y 

#读取数据
data_train = np.genfromtxt("hw2_train.dat")
data_test = np.genfromtxt("hw2_test.dat")

#预处理数据
X_train, y_train = preprocess(data_train)
X_test, y_test = preprocess(data_test)

#Problem 19
s, theta, d, Ein = multi_decision_stump(X_train, y_train)
print(s, theta, d, Ein)

#Problem 20
n = X_test.shape[0]
Eout = np.sum(s * np.sign(X_test[:, d] - theta) != y_test) / n
print(Eout)
