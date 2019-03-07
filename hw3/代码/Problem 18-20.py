# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 18:14:03 2019

@author: qinzhen
"""

import numpy as np

def preprocess(X):
    """
    添加偏置项
    """
    n = X.shape[0]
    return np.c_[np.ones(n), X]

#数据读入
data_train = np.genfromtxt("hw3_train.dat")
X_train = data_train[:, :-1]
y_train = data_train[:, -1].reshape(-1, 1)
X_train = preprocess(X_train)
data_test = np.genfromtxt("hw3_test.dat")
X_test = data_test[:, :-1]
y_test = data_test[:, -1].reshape(-1, 1)
X_test = preprocess(X_test)

#定义函数
def sigmoid(s):
    return 1 / (np.exp(-s) + 1)

def gradient(X, w, y):
    temp1 = - X.dot(w) * y
    temp2 = sigmoid(temp1)
    temp3 = - X * y
    grad = np.mean(temp3 * temp2, axis=0).reshape(-1, 1)

    return grad

#数据组数和维度
n, m = X_train.shape

#Problem 18
w = np.zeros((m, 1))
k = 0.001

for i in range(2000):
    grad = gradient(X_train, w, y_train)
    w -= k * grad

#计算标签
y_test_pred = X_test.dot(w)
y_test_pred[y_test_pred > 0] = 1
y_test_pred[y_test_pred <= 0] = -1
#计算Eout
Eout = np.mean(y_test_pred != y_test)
#求出误差
print(Eout)
print(w)

#Problem 19
w = np.zeros((m, 1))
k = 0.01

for i in range(2000):
    grad = gradient(X_train, w, y_train)
    w -= k * grad

#计算标签
y_test_pred = X_test.dot(w)
y_test_pred[y_test_pred > 0] = 1
y_test_pred[y_test_pred <= 0] = -1
#计算Eout
Eout = np.mean(y_test_pred != y_test)
#求出误差
print(Eout)
print(w)

#Problem 20
w = np.zeros((m, 1))
k = 0.001

#计数器
j = 0
for i in range(2000):
    x = X_train[j, :].reshape(1, -1)
    s = gradient(x, w, y_train[j])
    w -= k * s
    #更新下标
    j += 1
    j = j % n

#计算标签
y_test_pred = X_test.dot(w)
y_test_pred[y_test_pred > 0] = 1
y_test_pred[y_test_pred <= 0] = -1
#计算sign(Xw)
Eout = np.mean(y_test_pred != y_test)
#求出误差
print(Eout)
print(w)
