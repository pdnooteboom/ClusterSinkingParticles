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
from scipy.spatial.distance import pdist, squareform
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

def readDinoset_stations(name):
    stations = read_csv(name, usecols=[0]).values[:,0]
    return stations

def readDinoset_species(name):
    species = read_csv(name, header=None, skip_blank_lines=True, index_col=False, 
                       encoding="utf-8", skipinitialspace=True,
                       dtype=str).values[:]
    return species[0]
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

#%%
def find_nearest_2a(lons, lats, lon, lat):
    # find the arguments where the elements of Flons and Flats are neirest in 
    # vLons and vLats
    dist = []
    dist = np.sqrt((lons-lon)**2+(lats-lat)**2)
    idl = np.argmin(dist)
    return lons[idl], lats[idl]

def find_nearest_args(vLons, vLats, Flats, Flons):
    # find the arguments where the elements of Flons and Flats are neirest in 
    # vLons and vLats
    args = []
    unlons = vLons
    unlats = vLats
    for i in range(len(Flons)):
        lon, lat = find_nearest_2a(unlons, unlats, Flons[i], Flats[i])
        args.append(np.where(np.logical_and(vLons==lon, vLats==lat))[0][0])
    return np.array(args)

def MDS_data(X, ndim=2):
    # Classical multidimensional scaling of matrix X
    X = X / X.sum(axis=1)[:,np.newaxis]
    print(np.min(np.sum(X, axis=1)))
    assert (np.isclose(np.sum(X, axis=1),1)).all()
    print(X.shape)
    D = pdist(X, metric='euclidean')
    n = X.shape[0]
    del X
    D2 = squareform(D**2)
    del D
    print("D**2 computed")
    
    H = np.eye(n) - np.ones((n, n))/n
    K = -H.dot(D2).dot(H)/2
    print("K computed")
    del D2
    vals, vecs = np.linalg.eigh(K)
    del K
    print("Done!")
    indices   = np.argsort(vals)[::-1]
    vals = vals[indices]
    vecs = vecs[:,indices]
    indices_relevant, = np.where(vals > 0)
    Xbar  = vecs[:,indices_relevant].dot(np.diag(np.sqrt(vals[indices_relevant])))
    return vals, Xbar[:,:ndim]
