# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 22:47:25 2018

@author: Administrator
"""

import numpy as np
from scipy.linalg import inv

data = np.genfromtxt('hw2_lssvm_all.dat')

#获得K
def generateK(X, X1, gamma):
    n = X.shape[0]
    m = X1.shape[0]
    K = np.zeros((n,m))
    for i in range(n):
        K[i, :] = - np.sum((X1 - X[i])**2, axis = 1)
    return np.exp(gamma*K)

n = int(data.shape[0] * 0.8)
m = data.shape[0] - n

trainx = data[:n,:][:, :-1]
trainy = data[:n,:][:, -1]
testx = data[n:,:][:, :-1]
testy = data[n:,:][:, -1]

Gamma = [32, 2, 0.125]
Lambda = [0.001, 1, 1000]

gammatrain = Gamma[0]
lambdatrain = Lambda[0]
gammatest = Gamma[0]
lambdatest = Lambda[0]
Ein = 1
Eout = 1

for i in Gamma:
    K = generateK(trainx, trainx, i)
    K1 = generateK(trainx, testx, i)
    for j in Lambda:
        beta = inv(np.eye(n)*j + K).dot(trainy)
        r1 = beta.T.dot(K)
        r2 = beta.T.dot(K1).T
        ein = np.sum(np.sign(r1) != trainy)/n
        eout = np.sum(np.sign(r2) != testy)/m
        if(ein < Ein):
            Ein = ein
            gammatrain = i
            lambdatrain = j
        if(eout < Eout):
            Eout = eout
            gammatest = i
            lambdatest = j

print("minimum Ein =", Ein)
print("minimum Eout =", Eout)