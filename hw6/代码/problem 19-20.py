# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:27:00 2019

@author: qinzhen
"""

import numpy as np
from scipy.linalg import inv

data = np.genfromtxt('hw2_lssvm_all.dat')

#获得K
def generateK(X1, X2, gamma):
    """
    返回X1，X2
    """
    d1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    d2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    dist = d1 + d2 - 2 * X1.dot(X2.T)
    K = np.exp(- gamma * dist)
    
    return K

n = int(data.shape[0] * 0.8)
m = data.shape[0] - n

#划分测试集训练集
trainx = data[:n,:][:, :-1]
trainy = data[:n,:][:, -1]
testx = data[n:,:][:, :-1]
testy = data[n:,:][:, -1]

#初始化参数
Gamma = [32, 2, 0.125]
Lambda = [0.001, 1, 1000]
#记录最优解
gammatrain = Gamma[0]
lambdatrain = Lambda[0]
gammatest = Gamma[0]
lambdatest = Lambda[0]
Ein = 1
Eout = 1

for i in Gamma:
    #计算核矩阵
    K = generateK(trainx, trainx, i)
    K1 = generateK(testx, trainx, i)
    for j in Lambda:
        #计算beta
        beta = inv(np.eye(n)*j + K).dot(trainy)
        #计算预测结果
        y1 = K.dot(beta)
        y2 = K1.dot(beta)
        ein = np.mean(np.sign(y1) != trainy)
        eout = np.mean(np.sign(y2) != testy)
        #更新最优解
        if(ein < Ein):
            Ein = ein
            gammatrain = i
            lambdatrain = j
        if(eout < Eout):
            Eout = eout
            gammatest = i
            lambdatest = j

#### Problem 19
print("minimum Ein =", Ein)
print("gamma =", gammatrain)
print("lambda =", lambdatrain)

#### Problem 20
print("minimum Eout =", Eout)
print("gamma =", gammatest)
print("lambda =", lambdatest)