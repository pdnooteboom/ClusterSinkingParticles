
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 14:43:37 2020

@author: nooteboom
"""

import os
assert os.environ['CONDA_DEFAULT_ENV']=='Cartopy-py3', 'You should use the Cartopy-py3 conda environment here'
import numpy as np
import network_functions as nwf
from scipy.spatial.distance import pdist, squareform
from time import time
from sklearn.cluster import  cluster_optics_xi, cluster_optics_dbscan
import matplotlib
from netCDF4 import Dataset


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def normalise(X):
    return X / np.max(X)

def normalise2(X):
    X = X - np.min(X)
    return X / np.max(X)

def standardize(X):
    return (X- np.mean(X)) / np.std(X)

def direct_embedding(lons, lats):
    r0 = 6371.
    a = np.pi/180.
    x = np.array([r0 * np.cos(a * la) * np.cos(a * lo)  for lo, la in zip(lons, lats)])
    y = np.array([r0 * np.cos(a * la) * np.sin(a * lo)  for lo, la in zip(lons, lats)])
    z = np.array([r0 * np.sin(a * la)  for la in lats])
    X = np.concatenate((x[:,np.newaxis],y[:,np.newaxis],z[:,np.newaxis]), axis=1)

    assert X.shape[0]==len(x)
    return X

def find_nearest_2a(lons, lats, lon, lat):
    dist = []
    dist = np.sqrt((lons-lon)**2+(lats-lat)**2)
    idl = np.argmin(dist)
    return lons[idl], lats[idl]

def find_nearest_args(vLons, vLats, Flats, Flons):
    args = []
    unlons = vLons
    unlats = vLats
    for i in range(len(Flons)):
        lon, lat = find_nearest_2a(unlons, unlats, Flons[i], Flats[i])
        args.append(np.where(np.logical_and(vLons==lon, vLats==lat))[0][0])
    return np.array(args)

def MDS_data(X, ndim=2):
    X = X / X.sum(axis=1)[:,np.newaxis]
    assert (np.isclose(np.sum(X, axis=1),1)).all()
    print(X.shape)
    print("Computing classical MDS embedding (takes a while!)")
    ti = time()
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
    print('total time (mins): ',(time()-ti)/60)
    return vals, Xbar[:,:ndim]

#%% vars

sp = 250
extend = 'SO'
maxlat = 5
xiss = np.arange(0.0001,0.008, 0.0001)
xiss = np.arange(0.008,0.01, 0.0001)
xiss = np.arange(0.0001,0.01, 0.0001)


#for mins in [100,200,300,400]:#,500,600,700,800,900,1000]: #
for mins in [100,200,300,400,500,600,700,800,900,1000]:#[800,900]:
    print('min:  %d'%(mins))
    for xis in xiss:
        opts = ["xi", xis]

        #%% Load the species data
        # Read Foram data
        readData = '/Volumes/HD/network_clustering/'
        data = nwf.readForamset(readData + 'ForamData.csv')
        Foramspecies = nwf.readForamset(readData + 'ForamDataHeader.txt')[0][21:]
        data[data=='N/A'] = '0'#'NaN'
        data = data[:, 21:].astype(np.float)
        Flats, Flons = nwf.readForamset_lonlats(readData + 'ForamData.csv')
        Flons[np.where(Flons<0)]+=360
        
        # Read Dino data
        FlonsDino, FlatsDino, Dinodata = nwf.readDinoset(readData+'dinodata_red.csv')
        FlonsDino[FlonsDino<0] += 360
        Dinodata[np.isnan(Dinodata)] = 0
        
        if(extend=='SO'):
            idx = np.where(Flats<maxlat)[0]
            Flons = Flons[idx]
            Flats = Flats[idx]
            data = data[idx]
            idx = np.where(FlatsDino<maxlat)[0]
            FlonsDino = FlonsDino[idx]
            FlatsDino = FlatsDino[idx]
            Dinodata = Dinodata[idx]
    
        #%%
        from scipy.interpolate import griddata
        
        dirRead = '/Users/nooteboom/Documents/GitHub/cluster_TM/cluster_SP/density/dens/matrices/WOAdata/'
        envs = ['temp', 'salt', 
                'P', 'N', 'Si']
        files = {
                'temp':'woa18_95A4_t00_01.nc',
                'salt':'woa18_95A4_s00_01.nc',
                'P':'woa18_all_p00_01.nc',
                'N':'woa18_all_n00_01.nc',
                'Si':'woa18_all_i00_01.nc',
                }
        varss = {
                'temp':'t_an',
                'salt':'s_an',
                'P':'p_an',
                'N':'n_an',
                'Si':'i_an',
                }
        Fenv = np.zeros((len(Flons), len(envs)))
        Dinoenv = np.zeros((len(FlonsDino), len(envs)))
        Fenv_nn = np.zeros((len(Flons), len(envs)))
        Dinoenv_nn = np.zeros((len(FlonsDino), len(envs)))
        
        for eni, en in enumerate(envs):
            ncf = Dataset(dirRead + files[en])
            var = ncf[varss[en]][0,0].flatten()
#            print(var)
            lon = ncf['lon'][:]
            lon[lon<0] += 360
            lat = ncf['lat'][:]
            lon, lat = np.meshgrid(lon, lat)
            lon = lon.flatten(); lat = lat.flatten()
            idx = (~var.mask)
            var = var[idx]; lon = lon[idx]; lat = lat[idx];
        #    var = standardize(var)#normalise(var)
            
            points = np.concatenate((lat[:,np.newaxis],lon[:,np.newaxis]), axis=1)
            
            res = griddata(points, var, (Flats, Flons), method='nearest')
            Fenv_nn[:,eni] = res
            res = normalise2(res)
            Fenv[:,eni] = res
            res = griddata(points, var, (FlatsDino, FlonsDino), method='nearest')
            Dinoenv_nn[:,eni] = res
            res = normalise2(res)
            Dinoenv[:,eni] = res
        
        #%% Load the clustering result
        dirr = '/Users/nooteboom/Documents/GitHub/cluster_TM/cluster_SP/density/dens/results/'
        ff = np.load(dirr+'OPTICS_sp%d_smin%d.npz'%(sp, mins))
        lon0 = ff['lon']
        lat0 = ff['lat']
        reach = ff['reachability']
        ordering = ff['ordering']
        predecessor = ff['predecessor']
        core_distances = ff['core_distances']
        
        m, c = opts[0], opts[1]
        if m == "xi":
            l, _ = cluster_optics_xi(reach, predecessor, ordering, mins, xi=c)
        else:
            l = cluster_optics_dbscan(reachability=reach,
                                                core_distances=core_distances,
                                               ordering=ordering, eps=c)
        labels = np.array([li % 20 if li>=0 else li for li in l])
        
        bounds = np.arange(-.5,np.max(l)+1.5,1)
        norms = matplotlib.colors.BoundaryNorm(bounds, len(bounds))
        
        
        args = find_nearest_args(lon0, lat0, Flats, Flons)
        Flabels = labels[args]
        
        args = find_nearest_args(lon0, lat0, FlatsDino, FlonsDino)
        Dinolabels = labels[args]
        
        #%%
        np.savez('loops/prep_CCA_sp%d_smin%d%s_%.5f'%(sp, mins, opts[0], opts[1]),
                 envnames = envs,
                 Dinolabels=Dinolabels, Dinoenv = Dinoenv, Dinodata=Dinodata,
                 Flabels=Flabels, Fenv = Fenv, data=data,
                 lon0 = lon0, lat0= lat0, 
                 Dinoenv_nn = Dinoenv_nn, Fenv_nn = Fenv_nn)
        


