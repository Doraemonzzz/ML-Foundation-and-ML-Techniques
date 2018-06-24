# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 07:20:27 2018

@author: Administrator
"""

import numpy as np
from numpy.linalg import inv

def E(u,v):
    return np.exp(u)+np.exp(2*v)+np.exp(u*v)+u*u-2*u*v+2*(v*v)-3*u-2*v

def partial(point):
    u=point[0]
    v=point[1]
    pu=np.exp(u)+v*np.exp(u*v)+2*u-2*v-3
    pv=2*np.exp(2*v)+u*np.exp(u*v)-2*u+4*v-2
    return np.array([pu,pv])

def dpartial(point):
    u=point[0]
    v=point[1]
    puu=np.exp(u)+np.exp(u*v)*(v**2)+2
    pvv=4*np.exp(2*v)+np.exp(u*v)*(u**2)+4
    puv=np.exp(u*v)*(1+u*v)-2
    return np.array([[puu,puv],[puv,pvv]])

point=np.zeros(2)
eta=0.01

for i in range(5):
    point-=inv(dpartial(point)).dot(partial(point))
    
print(E(point[0],point[1]))