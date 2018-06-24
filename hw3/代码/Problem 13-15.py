# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 07:21:42 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#产生n组点
def generate(n):
    data=[]
    for i in range(n):
        x=np.random.uniform(-1,1)
        y=np.random.uniform(-1,1)
        flag=np.sign(x*x+y*y-0.6)
        p=np.random.random()
        if (p<0.1):
            flag*=-1
        data.append([x,y,flag])
    return data
        
data=generate(1000)

x1=[i[0] for i in data if i[-1]>0]
y1=[i[1] for i in data if i[-1]>0]
x2=[i[0] for i in data if i[-1]<0]
y2=[i[1] for i in data if i[-1]<0]

plt.scatter(x1,y1,s=1)
plt.scatter(x2,y2,s=1)
plt.show()

#Problem 13
from numpy.linalg import inv

Ein=np.array([])
for i in range(1000):
    data=generate(1000)
    X=np.array([[1]+i[:-1] for i in data])
    Y=np.array([i[-1] for i in data])

    w=inv(X.T.dot(X)).dot(X.T).dot(Y)

    error=np.sum(np.sign(X.dot(w)*Y)<0)/1000
    Ein=np.append(Ein,error)

print(np.average(Ein))
plt.hist(Ein)
plt.title('Ein without feature transform')
plt.show()

#Problem 14
W=[]
Eout=np.array([])
for i in range(1000):
    data=generate(1000)
    X=np.array([[1]+i[:-1]+[i[0]*i[1],i[0]**2,i[1]**2] for i in data])
    Y=np.array([i[-1] for i in data])
    
    w=inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    #测试数据
    data1=generate(1000)
    X1=np.array([[1]+i[:-1]+[i[0]*i[1],i[0]**2,i[1]**2] for i in data1])
    Y1=np.array([i[-1] for i in data1])

    error=np.sum(np.sign(X1.dot(w)*Y1)<0)/1000
    Eout=np.append(Eout,error)
    
    #记录w
    W.append(w)

W=np.array(W)
w3=np.array([i[3] for i in W])
plt.hist(w3)
plt.title('w3')
plt.show()

print(W.mean(axis=0))
print("w3的平均值"+str(w3.mean()))

#Problem 15
plt.hist(Eout)
plt.title('Eout with feature transform')
plt.show()
print(Eout.mean())