    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:25:14 2020

@author: nooteboom
"""
import os
assert os.environ['CONDA_DEFAULT_ENV']=='Cartopy-py3', 'You should use the Cartopy_py3 conda environment here'
import numpy as np
import network_functions as nwf
from scipy.spatial.distance import pdist, squareform
from time import time
from sklearn.cluster import  cluster_optics_xi, cluster_optics_dbscan
from sklearn.metrics.pairwise import euclidean_distances
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
for mins in [50,100, 200, 300, 400, 500, 600,700,800,900, 1000]:
    for xis in [10.]:#np.arange(0.0001,0.008, 0.00025):
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
            idx = np.where(Flats<-10)[0]
            Flons = Flons[idx]
            Flats = Flats[idx]
            data = data[idx]
            idx = np.where(FlatsDino<-10)[0]
            FlonsDino = FlonsDino[idx]
            FlatsDino = FlatsDino[idx]
            Dinodata = Dinodata[idx]
            
        #%% The dinstance matrix based on taxonomy
        
        #Fvalue, XF = MDS_data(data)
        #Ftaxdist = euclidean_distances(XF)
        #Dinovalue, XDino = MDS_data(Dinodata)
        #Dinotaxdist = euclidean_distances(XDino)
        
        Ftaxdist = euclidean_distances(data)
        Dinotaxdist = euclidean_distances(Dinodata)
        Ftaxdist = normalise(Ftaxdist)
        Dinotaxdist = normalise(Dinotaxdist)
        
        #%% The distance matrix based on the distance betweeen sites
        
        Fsitedist = direct_embedding(Flons, Flats)
        Fsitedist = euclidean_distances(Fsitedist)
        Dinositedist = direct_embedding(FlonsDino, FlatsDino)
        Dinositedist = euclidean_distances(Dinositedist)
        
        Fsitedist = np.round(normalise(Fsitedist),4)
        Dinositedist = np.round(normalise(Dinositedist),4)
        
        assert check_symmetric(Fsitedist, 0, 0)
        assert check_symmetric(Dinositedist, 0, 0)
        #%% The distance matrix based on the environmental variables
        from scipy.interpolate import griddata
        
        dirRead = '/Users/nooteboom/Documents/GitHub/cluster_TM/cluster_SP/density/dens/matrices/WOAdata/'
        envs = ['temp', 'salt', 'P', 'N', 'Si']
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
        
        for eni, en in enumerate(envs):
            ncf = Dataset(dirRead + files[en])
            var = ncf[varss[en]][0,0].flatten()
            lon = ncf['lon'][:]
            lon[lon<0] += 360
            lat = ncf['lat'][:]
            lat, lon = np.meshgrid(lat, lon)
            lon = lon.flatten(); lat = lat.flatten()
            idx = (~var.mask)
            var = var[idx]; lon = lon[idx]; lat = lat[idx];
        #    var = standardize(var)#normalise(var)
            
            points = np.concatenate((lat[:,np.newaxis],lon[:,np.newaxis]), axis=1)
            
            res = griddata(points, var, (Flats, Flons), method='nearest')
            res = normalise2(res)
            Fenv[:,eni] = res
            res = griddata(points, var, (FlatsDino, FlonsDino), method='nearest')
            res = normalise2(res)
            Dinoenv[:,eni] = res
        
        
        Dinoenvdist = euclidean_distances(Dinoenv)
        Fenvdist = euclidean_distances(Fenv)    
        assert check_symmetric(Fenvdist)
        assert check_symmetric(Dinoenvdist)
        #%% The distance matrix based on the OPTICS clustering
        from numba import jit
        
        @jit(nopython=True)
        def dis_from_reach_jit(X, reachability, args):
            for i in range(len(args)):
                for j in range(len(args)):
                    if(args[i]<args[j]):
                        X[i,j] = max(reachability[args[i]:args[j]]) - min(reachability[args[i]:args[j]])
                        X[j,i] = X[i,j]   
            return X
        
        def dis_from_reach(reachability, args):
            X = np.zeros((len(args),len(args)))
            X = dis_from_reach_jit(X, reachability, args)
            return X
        
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
        labels = np.array([li for li in l])
        
        bounds = np.arange(-.5,np.max(l)+1.5,1)
        norms = matplotlib.colors.BoundaryNorm(bounds, len(bounds))
        
        
        lon0 = lon0[ordering][1:]
        lat0 = lat0[ordering][1:]
        reach = reach[ordering][1:]
        
        args = find_nearest_args(lon0, lat0, Flats, Flons)
        lon0red = lon0[args]
        lat0red = lat0[args]
        Flabels = labels[ordering][args]
        Freachdist = dis_from_reach(reach, args)
        Freachdist = normalise(Freachdist)
        
        args = find_nearest_args(lon0, lat0, FlatsDino, FlonsDino)
        lon0red = lon0[args]
        lat0red = lat0[args]
        Dinolabels = labels[ordering][args]
        Dinoreachdist = dis_from_reach(reach, args)
        Dinoreachdist = normalise(Dinoreachdist)
        
        
        assert check_symmetric(Freachdist)
        assert check_symmetric(Dinoreachdist)
        
        #%% Save distance matrices
        
        np.savez('dms/distance_matrices_sp%d_smin%d_%s_%.5f'%(sp, mins, opts[0], opts[1]),
                 Fsitedist=Fsitedist, Ftaxdist=Ftaxdist, Freachdist=Freachdist, Fenvdist=Fenvdist,
                 Dinositedist=Dinositedist, Dinotaxdist=Dinotaxdist, Dinoreachdist=Dinoreachdist,
                 Dinoenvdist=Dinoenvdist,
                 Flabels=Flabels, Dinolabels=Dinolabels
                 )
        
        
