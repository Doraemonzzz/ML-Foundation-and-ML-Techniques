# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:11:34 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

####Problem 2

#原始图
x = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
z = np.array([-1, -1, -1, +1, +1, +1, +1])

x1 = x[z>0][:, 0]
y1 = x[z>0][:, 1]
x2 = x[z<0][:, 0]
y2 = x[z<0][:, 1]

plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.show()

#特征转换之后的图
def phi_1(x):
    return x[1] ** 2 - 2 * x[0] + 3

def phi_2(x):
    return x[0] ** 2 - 2 * x[1] - 3

X = []
for i in x:
    X.append([phi_1(i), phi_2(i)])
X = np.array(X)
    
X1 = X[z>0][:, 0]
Y1 = X[z>0][:, 1]
X2 = X[z<0][:, 0]
Y2 = X[z<0][:, 1]

plt.scatter(X1,Y1)
plt.scatter(X2,Y2)
plt.show()

#曲线图
y3 = np.arange(-2, 2, 0.01)
x3 = np.array([(i * i - 1.5) / 2 for i in y3])

plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.scatter(x3, y3, s=1)
plt.show()


####Problem3
clf = svm.SVC(kernel='poly', degree=2, coef0=1, gamma=1, C=1e10)
clf.fit(x, z)
alpha = z[clf.support_] * clf.dual_coef_[0]


####Problem4
b = clf.intercept_[0]
print(b)
print(x[clf.support_])
print(clf.dual_coef_[0])

####Problem5
#点的数量
n = 1000
r = 3

#作点
a = np.linspace(-r, r, n)
b = np.linspace(-r, r, n)

#构造网格
A, B = np.meshgrid(a, b)
X = np.c_[A.reshape(-1, 1), B.reshape(-1, 1)]
label = np.reshape(clf.predict(X), A.shape)

#绘制等高线
plt.contour(A, B, label, 0)

plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.show()