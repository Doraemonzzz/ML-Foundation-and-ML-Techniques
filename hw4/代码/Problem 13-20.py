# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:39:38 2018

@author: Administrator
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

####Problem 13
#读取数据
def read_data(file):
    x=[]
    y=[]
    with open(file) as f:
        for i in f.readlines():
            i=list(map(float,i.strip().split(' ')))
            temp=[1]
            temp+=i[:2]
            x.append(temp)
            y.append(i[-1])
    return np.array(x),np.array(y)

#计算w
def w(X,Y,k):
    N=X.shape[1]
    w=inv(X.T.dot(X)+k*np.eye(N)).dot(X.T).dot(Y)
    return w

#计算误差
def E(X,Y,w):
    N=X.shape[0]
    return np.sum(np.sign(X.dot(w))!=Y)/N

Xtrain,Ytrain=read_data('hw4_train.dat')
Xtest,Ytest=read_data('hw4_test.dat')

w1=w(Xtrain,Ytrain,11.26)
Ein=E(Xtrain,Ytrain,w1)
Eout=E(Xtest,Ytest,w1)
print("Ein="+str(Ein))
print("Eout="+str(Eout))


####Problem 14,15
K=range(-10,3)
Ein=[]
Eout=[]
for k in K:
    l=10**(k)
    w1=w(Xtrain,Ytrain,l)
    ein=E(Xtrain,Ytrain,w1)
    eout=E(Xtest,Ytest,w1)
    Ein.append(ein)
    Eout.append(eout)
    
plt.plot(K,Ein,label='Ein')
plt.plot(K,Eout,label='Eout')
plt.xlabel('$\lambda$')
plt.title('Ein VS Eout')
plt.legend()
plt.show()


####Problem 16,17
X1=Xtrain[:120,:]
Y1=Ytrain[:120]
Xval=Xtrain[120:,:]
Yval=Ytrain[120:]

Ein=[]
Eout=[]
Eval=[]
for k in K:
    l=10**(k)
    w1=w(X1,Y1,l)
    ein=E(X1,Y1,w1)
    eout=E(Xtest,Ytest,w1)
    eva=E(Xval,Yval,w1)
    Ein.append(ein)
    Eout.append(eout)
    Eval.append(eva)
    
plt.plot(K,Ein,label='Etrain')
plt.plot(K,Eout,label='Eout')
plt.plot(K,Eval,label='Eval')
plt.xlabel('$\lambda$')
plt.title('Ein VS Eout VS Eval')
plt.legend()
plt.show()


####Problem 16,18
w1=w(Xtrain,Ytrain,1)
Ein=E(Xtrain,Ytrain,w1)
Eout=E(Xtest,Ytest,w1)
print("Ein="+str(Ein))
print("Eout="+str(Eout))


####Problem 19
#准备数据
N=Xtrain.shape[0]//5
data=[]
for i in range(5):
    xtrain=np.concatenate((Xtrain[:i*N],Xtrain[(i+1)*N:]))
    ytrain=np.concatenate((Ytrain[:i*N],Ytrain[(i+1)*N:]))
    xval=Xtrain[i*N:(i+1)*N]
    yval=Ytrain[i*N:(i+1)*N]
    data.append([xtrain,ytrain,xval,yval])

Ecv=[]
K=range(-10,3)
for k in K:
    l=10**(k)
    ecv=0
    for i in data:
        xtrain=i[0]
        ytrain=i[1]
        xval=i[2]
        yval=i[3]
        w1=w(xtrain,ytrain,l)
        ecv+=E(xval,yval,w1)
    ecv/=5
    Ecv.append(ecv)
    
plt.plot(K,Ecv,label='Ecv')
plt.xlabel('$\lambda$')
plt.title('Ecv')
plt.legend()
plt.show()

####Problem20
k=10**(-8)
w1=w(Xtrain,Ytrain,k)
Ein=E(Xtrain,Ytrain,w1)
Eout=E(Xtest,Ytest,w1)
print("Ein="+str(Ein))
print("Eout="+str(Eout))
    
    
