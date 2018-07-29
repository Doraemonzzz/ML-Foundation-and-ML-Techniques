# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 10:54:00 2018

@author: Administrator
"""

####Problem 15
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

def transformdata(file):
    X = []
    Y = []
    with open(file) as data:
        for i in data.readlines():
            j = list(map(float,i.strip().split()))
            y = j[0]
            x = j[1:]
            Y.append(y)
            X.append(x)
    return np.array(X),np.array(Y)

train = "featurestrain.txt"
X_train,Y_trian = transformdata(train)
test = "featurestest.txt"
X_test,Y_test = transformdata(test)

#作图
x1 = X_train[Y_trian==0][:,0]
y1 = X_train[Y_trian==0][:,1]
x2 = X_train[Y_trian!=0][:,0]
y2 = X_train[Y_trian!=0][:,1]

plt.scatter(x1,y1,s=1,label = '0')
plt.scatter(x2,y2,s=1,label = 'not 0')
plt.title('0 VS not 0')
plt.legend()
plt.show()

#训练模型并作图
Y_trian1 = (Y_trian==0)
C = [-6,-4,-2,0,2]
W = []

for i in C:
    c = 10**i
    clf = svm.SVC(kernel = "linear",C = c)
    clf.fit(X_train,Y_trian1)
    w = clf.coef_[0]
    W = np.append(W,np.sqrt(np.sum(w*w)))

plt.plot(C,W)
plt.title("$||w||$ vs $log_{10}C$")
plt.show()

####Problem 16
x3 = X_train[Y_trian==8][:,0]
y3 = X_train[Y_trian==8][:,1]

x4 = X_train[Y_trian!=8][:,0]
y4 = X_train[Y_trian!=8][:,1]

plt.scatter(x3,y3,s=1,label = '8')
plt.scatter(x4,y4,s=1,label = 'not 8')
plt.title('8 VS not 8')
plt.legend()
plt.show()

#训练
Y_trian2 = 2*(Y_trian==8)-1
C = [-6,-4,-2,0,2]
Ein = []
alpha = []
for i in C:
    c = 10**i
    clf = svm.SVC(kernel='poly',degree=2,coef0=1,gamma=1,C = c)
    clf.fit(X_train,Y_trian2)
    e = np.sum(clf.predict(X_train) != Y_trian2)/len(X_train)
    support = clf.support_
    coef = np.sum(clf.dual_coef_[0]*Y_trian2[support])
    alpha.append(coef)
    Ein.append(e)

#作图
plt.plot(C,Ein)
plt.show()

####Problem 17
plt.plot(C,alpha)
plt.show()

####Problem 18
C = [-3,-2,-1,0,1]
Distance = []

for i in C:
    c = 10**i
    clf = svm.SVC(kernel='rbf',gamma=1,C = c)
    clf.fit(X_train,Y_trian1)
    w = clf.dual_coef_[0].dot(clf.support_vectors_)
    distance = 1/np.sum(w*w)
    Distance.append(distance)
    
plt.plot(C, Distance)
plt.show()

####Problem 19
Y_test1 = (Y_test==0)

Gamma = range(5)
Eout = []

for i in Gamma:
    gamma = 10**i
    clf = svm.SVC(kernel='rbf',gamma=gamma,C = 0.1)
    clf.fit(X_train,Y_trian1)
    e = np.sum(clf.predict(X_test) != Y_test1)/len(X_test)
    Eout.append(e)
    
plt.plot(Gamma,Eout)
plt.show()

####Problem 20
'''
from sklearn.model_selection import train_test_split

#对数据合并，方便调用train_test_split函数
Data = np.concatenate((X_train,Y_trian1.reshape(-1,1)),axis=1)
N = 100
Cnt = np.zeros(5)
Gamma = range(5)

for _ in range(N):
    train_set, val_set = train_test_split(Data, test_size=0.2)
    #取特征
    Xtrain = train_set[:,:2]
    #取标签
    Ytrain = train_set[:,2]
    Xval = val_set[:,:2]
    Yval = val_set[:,2]
    Eval = np.array([])

    for i in Gamma:
        gamma = 10**i
        clf = svm.SVC(kernel='rbf',gamma=gamma,C = 0.1)
        clf.fit(Xtrain,Ytrain)
        e = np.sum(clf.predict(Xval) != Yval)/len(Xval)
        Eval = np.append(Eval,e)
    index = np.argmin(Eval)
    Cnt[index] += 1

plt.bar(Gamma,Cnt)
plt.show()
'''

