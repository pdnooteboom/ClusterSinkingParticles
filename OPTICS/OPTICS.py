#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:14:08 2020

This script calculates adn writes a file with the reachability and the
ordering of points.

@author: nooteboom
"""
import os
assert os.environ['CONDA_DEFAULT_ENV']=='Cartopy-py3', 'You should use the conda environment Cartopy-py3'
import numpy as np
from netCDF4 import Dataset
from sklearn.cluster import OPTICS

def direct_embedding(lons, lats):
    r0 = 6371.
    a = np.pi/180.
    x = np.array([r0 * np.cos(a * la) * np.cos(a * lo)  for lo, la in zip(lons, lats)])
    y = np.array([r0 * np.cos(a * la) * np.sin(a * lo)  for lo, la in zip(lons, lats)])
    z = np.array([r0 * np.sin(a * la)  for la in lats])
    X = np.hstack((x, y, z))
    
    return X

def create_locs(latss, lonss, vLons, vLats, mask):
    endt = 122*1 # The amount of particles per location
    locs = [] # the locations which are not land
    lats = []
    lons = []
    lat0 = []
    lon0 = []
    n0 = 0
    for i in range(len(latss)):
        if(~mask[i].all()):
            if(len(latss[i][np.logical_and(~mask[i],latss[i]!=0)][:endt])==endt):
                locs.append(i)
                lats.append(latss[i][np.logical_and(~mask[i],latss[i]!=0)][:endt])
                lons.append(lonss[i][np.logical_and(~mask[i],latss[i]!=0)][:endt])
                lat0.append(vLats[i])
                lon0.append(vLons[i])
            else:
                n0 += 1
    print(n0 / len(lats))
    return lons, lats, lon0, lat0

readdir = '/Volumes/HD/network_clustering/PT/NEMO/'
readdir = '/Volumes/HD/network_clustering/PT/'
sp = 50 # The sinking speed (m/day)
mins = 300 # The variable s_{min}, which is used by OPTICS

ncf = Dataset(readdir + 'timeseries_per_location_inclatlonadv_ddeg1_sp%d_dd10.nc'%(sp))

lats = ncf['lat'][:]
lons, lats, lon0, lat0 = create_locs(lats.data, ncf['lon'][:].data,ncf['vLons'][:].data,
                                     ncf['vLats'][:].data, lats.mask)
lons = np.array(lons); lats = np.array(lats); lon0= np.array(lon0); lat0 = np.array(lat0)

  
X = direct_embedding(lons, lats)
#%%
P = {}
P = {
    "MinPts": mins,
    "optics_params": [["dbscan", 4000],
                      ["xi", 0.002],
                      ],
    "ylims": [200, 20000]}

#%%
optics_clustering = OPTICS(min_samples=mins, metric="euclidean").fit(X)
reachability = optics_clustering.reachability_
core_distances = optics_clustering.core_distances_
ordering = optics_clustering.ordering_
predecessor = optics_clustering.predecessor_

#%%
np.savez('results/OPTICS_sp%d_smin%d'%(sp,mins), reachability = reachability,
         core_distances=core_distances, ordering=ordering, predecessor=predecessor,
         lon=lon0, lat=lat0)
