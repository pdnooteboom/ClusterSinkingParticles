#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:15:35 2020

@author: nooteboom
"""
import numpy as np
from pandas import read_csv
#%% To read datasets
def readForamset(name):
    # function reads the foraminifera dataset
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
    # function reads the foraminifera dataset
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
    # function reads the dinocyst dataset
    lons = read_csv(name, usecols=[0]).values[:,0]
    lats = read_csv(name, usecols=[1]).values[:,0]
    data = read_csv(name).values[:,2:]
    data[np.isnan(data)] = 0
    row_sum = data.sum(axis=1)
    data = data / row_sum[:,np.newaxis]
    
    return lons, lats, data

def find_nearest_2a(lons, lats, lon, lat):
    # find grid locations near locations of sedimentary data locations
    dist = []
    dist = np.sqrt((lons-lon)**2+(lats-lat)**2)
    idl = np.argmin(dist)
    return lons[idl], lats[idl]

def find_nearest_args(vLons, vLats, Flats, Flons):
    # find arguments of grid locations near locations of sedimentary data locations
    args = []
    unlons = vLons
    unlats = vLats
    for i in range(len(Flons)):
        lon, lat = find_nearest_2a(unlons, unlats, Flons[i], Flats[i])
        args.append(np.where(np.logical_and(vLons==lon, vLats==lat))[0][0])
    return np.array(args)

def normalise(X):
    # normalise X
    return X / np.max(X)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    # Check if a is symmetric
    return np.allclose(a, a.T, rtol=rtol, atol=atol)