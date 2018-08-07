# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 22:41:47 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

train = np.genfromtxt('hw2_adaboost_train.dat')
test = np.genfromtxt('hw2_adaboost_test.dat')
plt.scatter(train[:, 0][train[:, 2] == 1], train[:, 1][train[:, 2] == 1])
plt.scatter(train[:, 0][train[:, 2] == -1], train[:, 1][train[:, 2] == -1])
plt.show()

#按第一个下标排序
train1 = np.array(sorted(train, key=lambda x:x[0]))

#按第二个下标排序
train2 = np.array(sorted(train, key=lambda x:x[1]))

#获得临界点
x1 = train1[:, 0]
threshold1 = np.append(np.array(x1[0]-0.1), (x1[:-1] + x1[1:])/2)
threshold1 = np.append(threshold1, x1[-1]+0.1)

x2 = train1[:, 1]
threshold2 = np.append(np.array(x2[0]-0.1), (x2[:-1] + x2[1:])/2)
threshold2 = np.append(threshold2, x2[-1]+0.1)

threshold = [threshold1, threshold2]

y = train1[:, 2 ]

n = len(train)

def decision_stump(X, U, threshold):
    #获得数据
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = X[:, 2]
    
    #获得数据数量
    n = len(x1)

    #记录维度
    d = 0
    #记录索引
    index = 0
    #记录Ein
    Ein = 1
    #记录s
    s = 1

    for i in range(n+1):
        t1 = threshold[0][i]
        #计算第一个维度的Ein
        E11 = (np.sign(x1 - t1) != y).dot(U)
        E12 = (np.sign(t1 - x1) != y).dot(U)
        if(E11 < Ein):
            d = 0
            index = i
            Ein = E11
            s = 1
        if(E12 < Ein):
            d = 0
            index = i
            Ein = E12
            s = -1
        #计算第二个维度的Ein
        t2 = threshold[1][i]
        E21 = (np.sign(x2 - t2) != y).dot(U)
        E22 = (np.sign(t2 - x2) != y).dot(U)
        if(E21 < Ein):
            d = 1
            index = i
            Ein = E21
            s = 1
        if(E22 < Ein):
            d = 1
            index = i
            Ein = E22
            s = -1
    return Ein,s,d,index

def Adaptive_Boosting(X,  threshold, T = 300):
    n = len(X)
    u = np.ones(n)/n

    #记录需要的数据
    Alpha = np.array([])
    U = np.array([])
    Epsilon = np.array([])
    Ein = np.array([])
    G = np.array([])

    #准备数据
    x1 = X[:, 0]
    x2 = X[:, 1]
    x = [x1, x2]
    y = X[:, 2]

    for t in range(T):
        ein,s,d,index = decision_stump(X, u, threshold)
        epsilon = u.dot((s*np.sign(x[d] - threshold[d][index])) != y)/np.sum(u)
        k = np.sqrt((1 - epsilon)/epsilon)
        #找到错误的点
        i1 = s*np.sign(x[d] - threshold[d][index]) != y
        #更新权重
        u[i1] = u[i1]*k
        #找到正确的点
        i2 = s*np.sign(x[d] - threshold[d][index]) == y
        #更新权重
        u[i2] = u[i2]/k
        alpha = np.log(k)
        
        #存储数据
        Ein = np.append(Ein, ein)
        if(t == 0):
            U = np.array([u])
        else:
            U = np.concatenate((U, np.array([u])),axis = 0)
        Epsilon = np.append(Epsilon, epsilon)
        Alpha = np.append(Alpha, alpha)
        g = [[s,d,index]]
        if(t == 0):
            G = np.array(g)
        else:
            G = np.concatenate((G,np.array(g)),axis = 0)
    return Ein, U, Epsilon, Alpha, G

#训练数据
Ein, U, Epsilon, Alpha, G = Adaptive_Boosting(train, threshold, T = 300)

#problem 12
T = 300
t = np.arange(T)

plt.plot(t, Ein)
plt.title("$E_{in}(g_t)\ VS\ t$")
plt.show()
print("Ein(g1) =", Ein[0], ",alpha1 =", Alpha[0])

#problem 14
def predeict(X, G, Alpha, t, threshold):
    "预测Ein(Gt)"
    x1 = X[:, 0]
    x2 = X[:, 1]
    x = [x1, x2]
    y = X[:, 2]
    N = len(X)
    
    s = G[:t, 0]
    d = G[:t, 1]
    thresh = G[:t, 2]
    alpha = Alpha[:t]

    result = []
    for i in range(t):
        s1 = s[i]
        d1 = d[i]
        t1 = thresh[i]
        #print(s1,d1,t1)
        result.append(s1*np.sign(x[d1] - threshold[d1][t1]))
    result = alpha.dot(np.array(result))
    
    
    return np.sum(np.sign(result) != y)/len(y)

T = 300
t = np.arange(T)
G1 = [predeict(train, G, Alpha, i, threshold) for i in t]

plt.plot(t, G1)
plt.title("$G_t\ VS\ t$")
plt.show()

print("Ein(G) =",G1[-1])

#problem 15
U1 = U.sum(axis = 1)

plt.plot(t,U1)
plt.title('$U_t$ VS t')
plt.show()

print("U2 =",U1[1],"UT =",U1[-1])

#problem 16
plt.plot(t,Epsilon)
plt.title('$\epsilon_t$ VS t')
plt.show()

print("minimun epsilon =",np.min(Epsilon))

#problem 17
x1 = test[:, 0]
x2 = test[:, 1]
xtest = [x1, x2]
ytest = test[:, 2]
N = len(x1)

s = G[:, 0]
d = G[:, 1]
thresh = G[:, 2]

g = []
for i in range(300):
    s1 = s[i]
    d1 = d[i]
    t1 = thresh[i]
    #print(s1,d1,t1)
    g.append(np.sum(s1*np.sign(xtest[d1] - threshold[d1][t1]) != ytest)/N)
    
plt.plot(t, g)
plt.title('$E_{out}(g_1)$ VS t')
plt.show()

print("Eout(g1) =",g[0])

#problem 18
T = 300
t = np.arange(T)
G2 = [predeict(test, G, Alpha, i, threshold) for i in t]

plt.plot(t, G2)
plt.title("$G_t\ VS\ t$")
plt.show()

print("Ein(G) =",G2[-1])

