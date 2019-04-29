# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:42:18 2019

@author: qinzhen
"""

####Problem 15
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

def transformdata(file):
    data = np.genfromtxt(file)
    y, X = data[:, 0], data[:, 1:]
    return X, y

train = "featurestrain.txt"
X_train, y_train = transformdata(train)
test = "featurestest.txt"
X_test,y_test = transformdata(test)

#作图
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], s=1, label='0')
plt.scatter(X_train[y_train != 0][:, 0], X_train[y_train != 0][:, 1], s=1, label='not 0')
plt.title('0 VS not 0')
plt.legend()
plt.show()


#训练模型并作图
y_train_1 = (y_train==0).astype("int")
C = [-6, -4, -2, 0, 2]
W = []

for i in C:
    c = 10 ** i
    clf = svm.SVC(kernel="linear", C=c)
    clf.fit(X_train, y_train_1)
    w = clf.coef_
    W.append(np.linalg.norm(w))

plt.plot(C, W)
plt.title("$||w||$ vs $log_{10}C$")
plt.show()

####Problem 16
plt.scatter(X_train[y_train == 8][:, 0], X_train[y_train == 8][:, 1], s=1, label='8')
plt.scatter(X_train[y_train != 8][:, 0], X_train[y_train != 8][:, 1], s=1, label='not 8')
plt.title('8 VS not 8')
plt.legend()
plt.show()

#训练
y_trian_8 = 2 * (y_train==8).astype("int") - 1
C = [-6, -4, -2, 0, 2]
Ein = []
alpha = []
for i in C:
    c = 10 ** i
    clf = svm.SVC(kernel='poly', degree=2, coef0=1, gamma=1, C=c)
    clf.fit(X_train, y_trian_8)
    e = np.mean(clf.predict(X_train) != y_trian_8)
    #支持向量的索引
    support = clf.support_
    #计算系数
    coef = np.sum(clf.dual_coef_[0] * y_trian_8[support])
    alpha.append(coef)
    Ein.append(e)

#作图
plt.plot(C, Ein)
plt.title("$\log_{10}C$ VS $E_{in}$")
plt.show()

####Problem 17
plt.plot(C, alpha)
plt.title("$\log_{10}C$ VS sum_alpha")
plt.show()

####Problem 18
C = [-3, -2, -1, 0, 1]
Distance = []
#将标签修改为-1, 1
y = 2 * y_train_1 - 1

for i in C:
    c = 10**i
    clf = svm.SVC(kernel='rbf', gamma=1, C=c)
    clf.fit(X_train, y)
    X = X_train[clf.support_]
    #距离矩阵
    d1 = np.sum(X ** 2, axis=1).reshape(-1, 1)
    d2 = np.sum(X ** 2, axis=1).reshape(1, -1)
    dist = d1 + d2 - 2 * X.dot(X.T)
    #Kernel矩阵
    K = np.exp(- c * dist)
    #计算anyn
    y1 = clf.dual_coef_[0] * y[clf.support_]
    w2 = y1.dot(K).dot(y1.T)
    #计算距离
    distance = 1 / np.sqrt(w2)
    Distance.append(distance)
    
plt.plot(C, Distance)
plt.title("$\log_{10}C$ VS distance")
plt.show()

####Problem 19
y_test_1 = (y_test == 0)

Gamma = range(5)
Eout = []

for i in Gamma:
    gamma = 10**i
    clf = svm.SVC(kernel='rbf', gamma=gamma, C=0.1)
    clf.fit(X_train, y_train_1)
    e = np.mean(clf.predict(X_test) != y_test_1)
    Eout.append(e)
    
plt.plot(Gamma, Eout)
plt.title("$\log_{10}\gamma$ VS $E_{out}$")
plt.show()

####Problem 20
from sklearn.model_selection import train_test_split

#对数据合并，方便调用train_test_split函数
Data = np.c_[X_train, y_train]
N = 100
#记录最小Eval对应的gamma的索引的次数
Cnt = np.zeros(5)
Gamma = range(5)

for _ in range(N):
    #划分数据
    train_set, val_set = train_test_split(Data, test_size=0.2)
    #取特征
    X_train = train_set[:, :2]
    #取标签
    y_train = train_set[:, 2]
    X_val = val_set[:, :2]
    y_val = val_set[:, 2]
    Eval = []

    for i in Gamma:
        gamma = 10**i
        clf = svm.SVC(kernel='rbf', gamma=gamma, C=0.1)
        clf.fit(X_train, y_train)
        e = np.mean(clf.predict(X_val) != y_val)
        Eval.append(e)
    #找到最小Eval对应的索引
    index = np.argmin(Eval)
    #对应索引次数加1
    Cnt[index] += 1

plt.bar(Gamma, Cnt)
plt.show()

