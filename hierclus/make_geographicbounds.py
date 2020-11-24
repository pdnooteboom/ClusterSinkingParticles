#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:43:07 2020

This scripts writes a .npy file which contains the information to plot the cluster 
edges (figure 2b)

@author: nooteboom
"""

import os
assert os.environ['CONDA_DEFAULT_ENV']=='Cartopy-py3', 'You should use the conda environment Cartopy-py3'
import numpy as np
import seaborn as sns
import network_functions as nwf

if(__name__=='__main__'):

    sns.set()
    sp=25
    exte = [1, 360, -75, 75]
    K = 150# 600 # number of eigenvectors computed
    L = 150 # 600 # number of clusters
    its = 150 # number of iterations used
    
    # Load the hierarchical clustering objects
    dirRead = '/Volumes/HD/network_clustering/clusteroutput/'
    dat = np.load(dirRead+'hier_clus_sp%d_exte%d_%d_%d_%d_K%d_L%d.npz'%(sp,exte[0],
                                                               exte[1],
                                                               exte[2],
                                                               exte[3],K,L), allow_pickle=True)
    
    networks = np.load(dirRead+'hier_clus_sp%d_exte%d_%d_%d_%d_K%d_L%d.npy'%(sp,exte[0],
                                                               exte[1],
                                                               exte[2],
                                                               exte[3],K,L), allow_pickle=True)
    
    #%%
    GB = nwf.Geographic_bounds(dat, networks)
    GB.create(its)
    
        #%%   
    np.savez('res/cluster_bounds_%dits_sp%d'%(its,sp),
             lons=GB.lons, lats=GB.lats,directions=GB.directions)
    
