#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
assert os.environ['CONDA_DEFAULT_ENV']=='skbio_env', 'You should use the conda environment skbio_env'
import numpy as np
import hierarchical_ANOSIM_functions as nwf
from sklearn.metrics.pairwise import euclidean_distances
from skbio import DistanceMatrix
from skbio.stats.distance import anosim

if(__name__=='__main__'):
    sp = 6 # sinking speed (m/day)
    maxlat = 65 # maximum latitude used for the sedimentary microplankton data
    season = 'summer'
    if(sp==6 and season==''):
        K=250
    else:
        K=150
    exte = [1, 360, -75, 75]
    iterations = 90
    perm = 999 # Amount of permutations used for the ANOSIM test
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
    
    # Reduce data to everything below 'maxlat' degrees North
    idx = np.where(Flats<maxlat)[0]
    Flons = Flons[idx]
    Flats = Flats[idx]
    data = data[idx]
    idx = np.where(FlatsDino<maxlat)[0]
    FlonsDino = FlonsDino[idx]
    FlatsDino = FlatsDino[idx]
    Dinodata = Dinodata[idx]
    
    #%% Load the clusters
    dirRead = '/Volumes/HD/network_clustering/clusteroutput/'
    
    if(season==''):
        dat = np.load(dirRead+'hier_clus_sp%d_exte%d_%d_%d_%d_K%d_L%d.npz'%(sp, exte[0], exte[1], exte[2],
                      exte[3], K, K),
                      allow_pickle=True)
    
        networks = np.load(dirRead+'hier_clus_sp%d_exte%d_%d_%d_%d_K%d_L%d.npy'%(sp,
                                                  exte[0],
                                                  exte[1],
                                                  exte[2],
                                                  exte[3], K, K),
                           allow_pickle=True)
    else:
        dat = np.load(dirRead+'hier_clus_sp%d_exte%d_%d_%d_%d_K%d_L%d_%s.npz'%(sp, exte[0], exte[1], exte[2],
                      exte[3], K, K, season),
                      allow_pickle=True)
    
        networks = np.load(dirRead+'hier_clus_sp%d_exte%d_%d_%d_%d_K%d_L%d_%s.npy'%(sp,
                                                  exte[0],
                                                  exte[1],
                                                  exte[2],
                                                  exte[3], K, K, season), allow_pickle=True)

    vLats = dat['vLats']; vLons= dat['vLons'];
    #%% The distance matrix based on taxonomy
    Ftaxdist = euclidean_distances(data)
    Dinotaxdist = euclidean_distances(Dinodata)
    Ftaxdist = np.round(nwf.normalise(Ftaxdist), 4)
    Dinotaxdist = np.round(nwf.normalise(Dinotaxdist), 4)
    assert nwf.check_symmetric(Ftaxdist)
    assert nwf.check_symmetric(Dinotaxdist)
    #%%
    DinoP = np.full(iterations, np.nan)
    FP = np.full(iterations, np.nan)
    DinoR = np.full(iterations, np.nan)
    FR = np.full(iterations, np.nan)
    for its in range(2,iterations):
        labelsg = np.arange(its).astype(str).tolist()
        field_plot = np.ones(dat['field_plot'].shape)*(-10000)
        labels=labelsg[:its];
        bounds = np.arange(-0.5,its+0.5,1)
        
        for k in range(its): field_plot[networks[its-1][k].cluster_indices] = networks[its-1][k].cluster_label
        field_plot = np.ma.masked_array(field_plot, field_plot==-10000)   
        
        #%% Determine from which clusters the data is part of:
        nomask = np.where(~field_plot.mask)
        field_plot = field_plot[nomask]
        args = nwf.find_nearest_args(vLons[nomask], vLats[nomask], Flats, Flons)
        Flabels = field_plot[args]
        args = nwf.find_nearest_args(vLons[nomask], vLats[nomask], FlatsDino, FlonsDino)
        Dinolabels = field_plot[args]
        #%%
        if(len(np.unique(Dinolabels))>1):
            Dano = anosim(DistanceMatrix(Dinotaxdist), Dinolabels.astype(str), permutations=perm)    
            DinoP[its] = list(Dano)[5]
            DinoR[its] = list(Dano)[4]
        if(len(np.unique(Flabels))>1):
            Fano = anosim(DistanceMatrix(Ftaxdist), Flabels.astype(str), permutations=perm)    
            FP[its] = list(Fano)[5]
            FR[its] = list(Fano)[4]
            
    #%% Save file with ANOSIM results
    np.savez('ANOSIM_hierarchicalclus%s_sp%d_perm%d_its%d_mlat%d.npz'%(season,
                                                                       sp,
                                                                       perm,
                                                                       iterations,
                                                                       maxlat),
             ForamP = FP, DinoP = DinoP,
             ForamR = FR, DinoR = DinoR)