# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:24:00 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt
X = np.genfromtxt("hw8_nolabel_train.dat")

class KMeans_():
    def __init__(self, k, D=1e-5):
        #聚类数量
        self.k = k
        #聚类中心
        self.cluster_centers_ = []
        #聚类结果
        self.labels_ = []
        #设置阈值
        self.D = D
        
    def fit(self, X):
        #数据维度
        n, d = X.shape
        #聚类标签
        labels = np.zeros(n, dtype=int)
        #初始中心点
        index = np.random.randint(0, n, self.k)
        cluster_centers = X[index]
        #记录上一轮迭代的聚类中心
        cluster_centers_pre = np.copy(cluster_centers)
        
        while True:
            #计算距离矩阵
            d1 = np.sum(X ** 2, axis=1).reshape(-1, 1)
            d2 = np.sum(cluster_centers ** 2, axis=1).reshape(1, -1)
            dist = d1 + d2 - 2 * X.dot(cluster_centers.T)
            
            #STEP1:找到最近的中心
            labels = np.argmin(dist, axis=1)
            #STEP2:重新计算中心
            for i in range(self.k):
                #第i类的索引
                index = (labels==i)
                #第i类的数据
                x = X[index]
                #判断是否有点和某聚类中心在一类
                if len(x) != 0:
                    cluster_centers[i] = np.mean(x, axis=0)
                
            #计算误差
            delta = np.linalg.norm(cluster_centers - cluster_centers_pre)
            
            if delta < self.D:
                break
            
            cluster_centers_pre = np.copy(cluster_centers)
            
        self.cluster_centers_ = np.copy(cluster_centers)
        self.labels_ = labels
        
        
    def predict(self, X):
        #计算距离矩阵
        d1 = np.sum(X ** 2, axis=1).reshape(-1, 1)
        d2 = np.sum(self.cluster_centers_ ** 2, axis=1).reshape(1, -1)
        dist = d1 + d2 - 2 * X.dot(self.cluster_centers_.T)
        
        #找到最近的中心
        self.cluster_centers_ = np.argmin(dist, axis=1)
        
        return self.cluster_centers_

n = X.shape[0]
K = [2, 4, 6, 8, 10]
Ein = []
for k in K:
    #训练模型
    kmeans = KMeans_(k)
    kmeans.fit(X)
    #获得标签
    label = kmeans.labels_
    #获得聚类中心
    center = kmeans.cluster_centers_
    #计算Ein
    ein = 0
    for i in range(k):
        #计算每一类的误差
        ein += np.sum((X[label==i] - center[i]) ** 2)
    #计算均值
    ein /= n
    Ein.append(ein)
    
plt.scatter(K, Ein)
plt.title("$K$ VS $E_{in}$")
plt.xlabel("$K$")
plt.ylabel("$E_{in}$")
plt.show()


