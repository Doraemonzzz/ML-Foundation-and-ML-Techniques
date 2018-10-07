# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 00:22:33 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

#构造类
class DTree:
    def __init__(self, node, theta, d, left, right):
        self.node = node
        #阈值
        self.theta = theta
        #维度
        self.d = d
        self.left = left
        self.right = right
    
    #判断是否都为一类
    def ispure(self):
        num = np.sum(self.node[:, 2] == 1)
        return num == 0 or num == len(self.node)

#读取数据
def readdata(file):
    Data = []
    with open(file) as data:
        for i in data.readlines():
            i.strip()
            Data.append(list(map(float,i.split())))
    return np.array(Data)

#读取数据并作图
train = readdata('hw7_train.dat')
test = readdata('hw7_test.dat')

#作图
plt.scatter(train[:, 0][train[:, 2] == -1], train[:, 1][train[:, 2] == -1])
plt.scatter(train[:, 0][train[:, 2] == 1], train[:, 1][train[:, 2] == 1])
plt.show()

#Gini index
def Gini(y):
    N = len(y)
    if(N == 0):
        return 1
    t = np.sum(y == -1)/ N
    return 1 - t**2 - (1 - t)**2

#定义impurty
def lossfunc(theta, data, d):
    '''
    d为数据的维度，theta为decision stump的阈值
    '''
    index1 = (data[:, d] < theta)
    index2 = (data[:, d] >= theta)
    Gini1 = Gini(data[index1][:, 2])
    Gini2 = Gini(data[index2][:, 2])
    return len(index1) * Gini1 + len(index2) * Gini2

#在两个维度上分别利用decision stump计算，找到损失函数的最小值，返回维度以及阈值
def branch(data):
    '''
    在两个维度上分别利用decision stump计算，找到损失函数的最小值，返回维度以及阈值
    '''
    train = data
    #记录最优阈值以及损失函数的最小值以及维度
    theta = 0
    error = 10000
    d = 0
    
    #根据第一个维度
    train = np.array(sorted(train, key = lambda x: x[0]))
    #计算decision stump的阈值
    segmentx = train[:, 0]
    #
    for i in segmentx:
        error1 = lossfunc(i, train, 0)
        if error1 < error:
            error = error1
            theta = i

    #根据第二个维度排序
    train = np.array(sorted(train, key = lambda x: x[1]))
    #计算decision stump的阈值
    segmenty = train[:, 1]
    for i in segmenty:
        error2 = lossfunc(i, train, 1)
        if error2 < error:
            error = error2
            theta = i
            d = 1
    return theta, d

#构造学习函数
def isstop(data):
    '''
    判断是否停止，有两种情形，一种是没有数据，另一种是所有数据都为一类
    '''
    n = len(data)
    num = np.sum(data[:, 2] == -1)
    return num == n or num == 0

def learntree(data):
    if isstop(data):
        return DTree(data[0][2], 0, 0, None, None)
    else:
        theta, d = branch(data)
        tree = DTree(None, theta, d, None, None)
        #划分数据
        leftdata = data[data[:, d] < theta]
        rightdata = data[data[:, d] >= theta]
        #学习左树
        leftTree = learntree(leftdata)
        #学习右树
        rightTree = learntree(rightdata)
        #返回
        tree.left = leftTree
        tree.right = rightTree
        return tree
    
#预测函数
def pred(tree, data):
    if tree.left == None and tree.right == None:
        return tree.node
    if data[tree.d] < tree.theta:
        return pred(tree.left, data)
    else:
        return pred(tree.right, data)
    
#计算误差
def error(Dtree, data):
    ypred = [pred(Dtree, i) for i in data]
    return 1 - np.sum(ypred == data[:, 2]) / len(data)

dtree = learntree(train)

#14
print(error(dtree, train))

#15
print(error(dtree, test))

#16
N = 300
Ein = np.array([])
tree = []
m, n = train.shape
for i in range(N):
    index = np.random.randint(0, m, (m))
    traindata = train[index, :]
    dtree = learntree(traindata)
    tree.append(dtree)
    Ein = np.append(Ein, error(dtree, train))
    
plt.hist(Ein)
plt.show()

#17
def random_forest_error(tree, data):
    '''
    利用前k个树计算结果
    '''
    Error = np.array([])
    N = len(tree)
    for i in range(N):
        E = []
        for j in range(1+i):
            #E = np.append(E, error(tree[j], train))
            E.append([pred(tree[j], k) for k in data])
        E = np.array(E)
        #0视为1
        ypred = np.sign(E.sum(axis = 0) + 0.5)
        error = 1 - np.sum(ypred == data[:, 2]) / len(data)
        Error = np.append(Error, error)
    return Error

Ein_G = random_forest_error(tree, train)

plt.plot(np.arange(1, N+1), Ein_G)
plt.show()

#18
Eout_G = random_forest_error(tree, test)

plt.plot(np.arange(1, N+1), Eout_G)
plt.show()

#19
def learntree_new(data):
    theta, d = branch(data)
    tree = DTree(None, theta, d, None, None)
    #划分数据
    leftdata = data[data[:, d] < theta]
    rightdata = data[data[:, d] >= theta]
    #左树
    k1 = np.sign(np.sum(leftdata[:, 2]) + 0.5)#+0.5是为了防止出现0
    leftTree = DTree(k1, None, None, None, None)
    #右树
    k2 = np.sign(np.sum(rightdata[:, 2]) + 0.5)
    rightTree = DTree(k2, None, None, None, None)
    #返回
    tree.left = leftTree
    tree.right = rightTree
    return tree

N = 500
newtree = []
m, n = train.shape
for i in range(N):
    index = np.random.randint(0, m, (m))
    traindata = train[index, :]
    dtree = learntree_new(traindata)
    newtree.append(dtree)
    
newEin_G = random_forest_error(newtree, train)

plt.plot(np.arange(1, N+1), newEin_G)
plt.show()

#20
Eout_G = random_forest_error(tree, test)

plt.plot(np.arange(1, N+1), Eout_G)
plt.show()