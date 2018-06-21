# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 07:13:50 2018

@author: Administrator
"""

import numpy as np

def E(u,v):
    return np.exp(u)+np.exp(2*v)+np.exp(u*v)+u*u-2*u*v+2*(v*v)-3*u-2*v

def partial(point):
    u=point[0]
    v=point[1]
    pu=np.exp(u)+v*np.exp(u*v)+2*u-2*v-3
    pv=2*np.exp(2*v)+u*np.exp(u*v)-2*u+4*v-2
    return np.array([pu,pv])

point=np.zeros(2)
eta=0.01

for i in range(5):
    point-=eta*partial(point)
    
print(E(point[0],point[1]))