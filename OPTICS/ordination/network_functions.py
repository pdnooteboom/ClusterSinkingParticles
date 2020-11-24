#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:15:35 2020

@author: nooteboom
"""
from numba import jit
import numpy as np
import math
from pandas import read_csv
#%% To read datasets
def readForamset(name):
    file = open(name)
    data = []
    line = file.readline()
    while(len(line)>0):
        ldat = []
        j = 0
        for l in range(len(line)):
            if(line[j:l][-1:] in ['\t','\n']):
                ldat.append(line[j:l-1])
                j = l
        line = file.readline()
        data.append(ldat)
        
    data = np.array(data)
    return data

def readForamset_lonlats(name):
    file = open(name)
    data = []
    line = file.readline()
    while(len(line)>0):
        tabcount = 0
        ldat = []
        j = 0
        for l in range(len(line)):
            if(line[j:l][-1:] in ['\t','\n']):
                tabcount+=1
                if(tabcount in [5,6]):
                    ldat.append(line[j:l-1])
                j = l
        line = file.readline()
        data.append(ldat)
        
    data = np.array(data)
    return data[:,0].astype(np.float), data[:,1].astype(np.float)

def readDinoset(name):
    lons = read_csv(name, usecols=[0]).values[:,0]
    lats = read_csv(name, usecols=[1]).values[:,0]
    data = read_csv(name).values[:,2:]
    data[np.isnan(data)] = 0
    row_sum = data.sum(axis=1)
    data = data / row_sum[:,np.newaxis]
    
    return lons, lats, data
#%% To calculate similarity measures

#@jit(nopython=True)
def Euclidean_distance(x,y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    if(res==0):
        res = 1 / math.sqrt(res / 4)
    return 1 / math.sqrt(res / 4)

@jit(nopython=True)
def Cos_sim(x,y):
    res = 0
    normx = 0
    normy = 0
    for i in range(len(x)):
        res += (x[i]* y[i])
        normx += x[i]**2
        normy += y[i]**2
    return ((res / math.sqrt(normx*normy)))#/2

@jit(nopython=True)
def Similarity_data(data):
    res = np.ones((data.shape[0],data.shape[0]))
    for i in range(res.shape[0]):
        for j in range(i):
             EC = Cos_sim(data[i],data[j])
             res[i,j] = EC
             res[j,i] = res[i,j]
    return res

#%%

@jit(nopython=True)
def get_edges(TM):
    res = []
    for i in range(TM.shape[0]):
        for j in range(TM.shape[1]):
            if(TM[i,j]>0):
                res.append((i,-j))
    return res

@jit(nopython=True)
def get_weighted_edges(TM):
    res = []
    for i in range(TM.shape[0]):
        for j in range(TM.shape[1]):
            res.append((i,-j, TM[i,j]))
    return res

