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
from sklearn.metrics.pairwise import euclidean_distances
from netCDF4 import Dataset

#%% vars
sp = 250 # sinking speed (m/day)
maxlat = 65 # use the sedimentary dat below maxlat deg N
for mins in [50,100, 200, 300, 400, 500, 600,700,800,900, 1000]: # s_min values
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
    
    # limit the data below maxlat
    idx = np.where(Flats<maxlat)[0]
    Flons = Flons[idx]
    Flats = Flats[idx]
    data = data[idx]
    idx = np.where(FlatsDino<maxlat)[0]
    FlonsDino = FlonsDino[idx]
    FlatsDino = FlatsDino[idx]
    Dinodata = Dinodata[idx]
        
    #%% The dinstance matrix based on taxonomy
    
    Ftaxdist = nwf.euclidean_distances(data)
    Dinotaxdist = nwf.euclidean_distances(Dinodata)
    Ftaxdist = nwf.normalise(Ftaxdist)
    Dinotaxdist = nwf.normalise(Dinotaxdist)
    
    #%% The distance matrix based on the distance betweeen sites
    
    Fsitedist = nwf.direct_embedding(Flons, Flats)
    Fsitedist = nwf.euclidean_distances(Fsitedist)
    Dinositedist = nwf.direct_embedding(FlonsDino, FlatsDino)
    Dinositedist = nwf.euclidean_distances(Dinositedist)
    
    Fsitedist = np.round(nwf.normalise(Fsitedist),4)
    Dinositedist = np.round(nwf.normalise(Dinositedist),4)
    
    assert nwf.check_symmetric(Fsitedist, 0, 0)
    assert nwf.check_symmetric(Dinositedist, 0, 0)
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
        
        points = np.concatenate((lat[:,np.newaxis],lon[:,np.newaxis]), axis=1)
        
        res = griddata(points, var, (Flats, Flons), method='nearest')
        res = nwf.normalise2(res)
        Fenv[:,eni] = res
        res = griddata(points, var, (FlatsDino, FlonsDino), method='nearest')
        res = nwf.normalise2(res)
        Dinoenv[:,eni] = res
    
    Dinoenvdist = euclidean_distances(Dinoenv)
    Fenvdist = euclidean_distances(Fenv)    
    assert nwf.check_symmetric(Fenvdist)
    assert nwf.check_symmetric(Dinoenvdist)
    #%% The distance matrix based on the OPTICS clustering
    
    dirr = '/Users/nooteboom/Documents/GitHub/cluster_TM/cluster_SP/density/dens/results/'
    ff = np.load(dirr+'OPTICS_sp%d_smin%d.npz'%(sp, mins))
    lon0 = ff['lon']
    lat0 = ff['lat']
    reach = ff['reachability']
    ordering = ff['ordering']
    predecessor = ff['predecessor']
    core_distances = ff['core_distances']
    
    lon0 = lon0[ordering][1:]
    lat0 = lat0[ordering][1:]
    reach = reach[ordering][1:]

    args = nwf.find_nearest_args(lon0, lat0, Flats, Flons)
    Freachdist = nwf.dis_from_reach(reach, args)
    Freachdist = nwf.normalise(Freachdist)
    
    args = nwf.find_nearest_args(lon0, lat0, FlatsDino, FlonsDino)
    Dinoreachdist = nwf.dis_from_reach(reach, args)
    Dinoreachdist = nwf.normalise(Dinoreachdist)
    
    
    assert nwf.check_symmetric(Freachdist)
    assert nwf.check_symmetric(Dinoreachdist)
    
    #%% Save distance matrices
    
    np.savez('dms/distance_matrices_sp%d_smin%d'%(sp, mins),
             Fsitedist=Fsitedist, Ftaxdist=Ftaxdist, Freachdist=Freachdist, Fenvdist=Fenvdist,
             Dinositedist=Dinositedist, Dinotaxdist=Dinotaxdist, Dinoreachdist=Dinoreachdist,
             Dinoenvdist=Dinoenvdist
             )