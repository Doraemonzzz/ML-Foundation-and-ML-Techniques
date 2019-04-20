# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:22:57 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt

#读取数据
train = np.loadtxt("hw8_train.dat")
test = np.loadtxt("hw8_test.dat")
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

class KNeighborsClassifier_():
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, X):
        #计算距离矩阵
        d1 = np.sum(X ** 2, axis=1).reshape(-1, 1)
        d2 = np.sum(self.X ** 2, axis=1).reshape(1, -1)
        dist = d1 + d2 - 2 * X.dot(self.X.T)
        
        #找到最近的k个点的索引
        index = np.argsort(dist, axis=1)[:, :self.n_neighbors]
        #计算预测结果
        y = np.sign(np.sum(self.y[index], axis=1))
        
        return y

# Q11-12
K = [1, 3, 5, 7, 9]
Ein = []
for k in K:
    #训练模型
    knn = KNeighborsClassifier_(n_neighbors=k)
    knn.fit(X_train, y_train)
    #预测结果
    y = knn.predict(X_train)
    ein = np.mean(y != y_train)
    Ein.append(ein)

plt.scatter(K, Ein)
plt.title("$K$ VS $E_{in}$")
plt.xlabel("$K$")
plt.ylabel("$E_{in}$")
plt.show()

# Q13-14
K = [1, 3, 5, 7, 9]
Eout = []
for k in K:
    #训练模型
    knn = KNeighborsClassifier_(n_neighbors=k)
    knn.fit(X_train, y_train)
    #预测结果
    y = knn.predict(X_test)
    eout = np.mean(y != y_test)
    Eout.append(eout)

plt.scatter(K, Eout)
plt.title("$K$ VS $E_{out}$")
plt.xlabel("$K$")
plt.ylabel("$E_{out}$")
plt.show()


class RBFNetworkClassifier():
    def __init__(self, gamma):
        self.gamma = gamma
        self.beta = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, X):
        #计算距离矩阵
        d1 = np.sum(X ** 2, axis=1).reshape(-1, 1)
        d2 = np.sum(self.X ** 2, axis=1).reshape(1, -1)
        dist = d1 + d2 - 2 * X.dot(self.X.T)
        
        #计算exp(-gamma*dist)
        d = np.exp(-self.gamma * dist)
        
        #计算预测结果
        y = np.sign(np.sum(d * self.y, axis=1))
        
        return y

# Q15-16
Gamma = [0.001, 0.1, 1, 10, 100]
Ein_rbf = []
for gamma in Gamma:
    #训练模型
    knn = RBFNetworkClassifier(gamma=gamma)
    knn.fit(X_train, y_train)
    #预测结果
    y = knn.predict(X_train)
    ein = np.mean(y != y_train)
    Ein_rbf.append(ein)

plt.scatter(Gamma, Ein_rbf)
plt.title("$\gamma$ VS $E_{in}$")
plt.xlabel("$\gamma$")
plt.ylabel("$E_{in}$")
plt.show()

# Q17-18
Gamma = [0.001, 0.1, 1, 10, 100]
Eout_rbf = []
for gamma in Gamma:
    #训练模型
    knn = RBFNetworkClassifier(gamma=gamma)
    knn.fit(X_train, y_train)
    #预测结果
    y = knn.predict(X_test)
    eout = np.mean(y != y_test)
    Eout_rbf.append(eout)

plt.scatter(Gamma, Eout_rbf)
plt.title("$\gamma$ VS $E_{out}$")
plt.xlabel("$\gamma$")
plt.ylabel("$E_{out}$")
plt.show()