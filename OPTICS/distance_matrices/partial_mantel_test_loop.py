#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:18:32 2020

@author: nooteboom
"""

import numpy as np
from ecopy import Mantel
import network_functions as nwf

def distm(ff, dist, name='F', ns=False):
    dist = ff[name+dist+'dist']
    dist = np.round(dist, 4)
    labels = ff[name+'labels']
    if(ns):
        args = np.where(labels!=-1)
        dist = dist[args]
        dist = dist[:,args][:,0]
    assert nwf.check_symmetric(dist, 0, 0), 'distm should be completely symmetric'
    
    return dist

np.random.seed(28) # seed for reproducibility of the permutations
sp = 25 # the sinking speed
extend = 'SO'
perm = 999 # amount of permutations used 
dist1 = 'reach'
dist2 = 'tax'
distc = 'site'
distances = ['env', 'site','reach']

#%%
no_noisy_sites = False

minss = [100, 200, 300, 400, 500, 600, 700,800,900, 1000]
# mantel test depends on distances only, not on clusters. using xiss is pointless
xiss = [10.]

FR = np.zeros((len(minss), len(xiss)))
DR = np.zeros((len(minss), len(xiss)))
FP = np.zeros((len(minss), len(xiss)))
DP = np.zeros((len(minss), len(xiss)))

for mini,mins in enumerate(minss):
    print(mins)
    for xii,xis in enumerate(xiss):
        opts = ["xi", xis ]
        dirr = '/Users/nooteboom/Documents/GitHub/cluster_TM/cluster_SP/density/dens/matrices/'
        ff = np.load(dirr+'dms/distance_matrices_sp%d_smin%d_%s_%.5f.npz'%(sp, mins, opts[0], opts[1]))

        Fdist1 = distm(ff, dist1, name='F')
        Ddist1 = distm(ff, dist1, name='Dino')
        Fdist2 = distm(ff, dist2, name='F')
        Ddist2 = distm(ff, dist2, name='Dino')
        Fdistc = distm(ff, distc, name='F')
        Ddistc = distm(ff, distc, name='Dino')

        
        #%% The partial mantel test       
        
        mant= Mantel(Fdist1, Fdist2, Fdistc, nperm=perm)
        coeff, p_value, n = mant.r_obs, mant.pval, mant.perm
        FR[mini, xii] = coeff
        FP[mini, xii] = p_value
        
        
        mant= Mantel(Ddist1, Ddist2, Ddistc, nperm=perm)
        coeff, p_value, n = mant.r_obs, mant.pval, mant.perm
        DR[mini, xii] = coeff
        DP[mini, xii] = p_value

#%%
np.savez('mantel_tests_sp%d_perm%d'%(sp,perm),
         DR=DR,DP=DP,
         FR=FR, FP=FP)
