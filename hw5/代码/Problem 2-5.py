# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 09:55:16 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

####Problem 2

#原始图
x = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
z = np.array([-1,-1,-1,+1,+1,+1,+1])

x1 = x[z>0][:,0]
y1 = x[z>0][:,1]
x2 = x[z<0][:,0]
y2 = x[z<0][:,1]

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.show()

#特征转换之后的图
def phi_1(x):
    return x[1]**2-2*x[0]+3

def phi_2(x):
    return x[0]**2-2*x[1]-3

X = []
for i in x:
    X.append([phi_1(i),phi_2(i)])
X = np.array(X)
    
X1 = X[z>0][:,0]
Y1 = X[z>0][:,1]
X2 = X[z<0][:,0]
Y2 = X[z<0][:,1]

plt.scatter(X1,Y1)
plt.scatter(X2,Y2)
plt.show()

#曲线图
y3 = np.arange(-2,2,0.01)
x3 = np.array([(i*i-1.5)/2 for i in y3])

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3,s=1)
plt.show()


####Problem3
clf = svm.SVC(kernel='poly',degree=2,coef0=1,gamma=1,C=1e10)
clf.fit(x,z)
alpha = z[clf.support_]*clf.dual_coef_[0]


####Problem4
def g(x):
    r = np.sqrt(2)
    return np.array([1,r*x[0],r*x[1],x[0]**2,x[0]*x[1],x[1]*x[0],x[1]**2])

support = clf.support_
coef = clf.dual_coef_[0]
x4 = np.array([g(i) for i in x])

#取第一个支持向量
s = support[0]

b = z[s] - coef.dot(x4[support].dot(x4[s]))
k = (coef).dot(x4[support])


####Problem5
#构造等高线函数
def g(x,y,k,b):
    r = np.sqrt(2)
    return k[0]+k[1]*r*x+k[2]*r*y+k[3]*(x**2)+(k[4]+k[5])*x*y+k[6]*(y**2)+b

#点的数量
n = 1000
r = 3

#作点
p = np.linspace(-r,r,n)
q = np.linspace(-r,r,n)

#构造网格
P,Q = np.meshgrid(p,q)

#绘制等高线
plt.contour(P,Q,g(P,Q,k,b),0)

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.show()