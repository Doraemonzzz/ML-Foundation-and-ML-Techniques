# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 07:32:40 2018

@author: Administrator
"""

import numpy as np

train=[]
test=[]

#数据读入
with open('hw3_train.dat') as file:
    for i in file.readlines():
        train.append([1]+list(map(float,i.strip().split(' '))))

with open('hw3_test.dat') as file:
    for i in file.readlines():
        test.append([1]+list(map(float,i.strip().split(' '))))
        
train=np.array(train)
test=np.array(test)

#定义函数
def f(y,w,x):
    temp=y*w.dot(x)
    return (-y*x)/(np.exp(temp)+1)

def sig(w,x):
    return 1/(math.exp(-w.dot(x))+1)

#数据维度
m=train.shape[1]-1
#数据组数
n=train.shape[0]

#Problem 18
w=np.zeros(m)
k=0.001

for i in range(2000):
    s=np.zeros(m)
    for j in range(n):
        s+=f(train[j][-1],w,train[j][:-1])
    s=s/n
    w-=k*s

#计算Xw
r1=test[:,:-1].dot(w)
#计算sign(Xw)
r2=np.sign(r1)
#求出误差
print((r2!=test[:,-1]).sum()/test.shape[0])
print(w)


#Problem 19
w=np.zeros(m)
k=0.01

for i in range(2000):
    s=np.zeros(m)
    for j in range(n):
        s+=f(train[j][-1],w,train[j][:-1])
    s=s/n
    w-=k*s

#计算Xw
r1=test[:,:-1].dot(w)
#计算sign(Xw)
r2=np.sign(r1)
#求出误差
print((r2!=test[:,-1]).sum()/test.shape[0])
print(w)

#Problem 20
w=np.zeros(m)
k=0.001

for i in range(2000):
    j=np.random.choice(n)
    s=f(train[j][-1],w,train[j][:-1])
    w-=k*s

#计算Xw
r1=test[:,:-1].dot(w)
#计算sign(Xw)
r2=np.sign(r1)
#求出误差
print((r2!=test[:,-1]).sum()/test.shape[0])
print(w)