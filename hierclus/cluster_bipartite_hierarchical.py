#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:07:26 2020

@author: nooteboom
"""
import os
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import network_functions as nwf
import cartopy.crs as ccrs
import seaborn as sns
from copy import copy
from scipy.sparse import csr_matrix
assert os.environ['CONDA_DEFAULT_ENV'] == 'Cartopy-py3', 'You should use the \
conda environment Cartopy-py3'

if(__name__ == '__main__'):
    colors = np.array(['midnightblue', 'dimgray', 'dimgray', 'dimgray',
                       'sienna', 'skyblue', 'dimgray',
                       'firebrick', 'dimgray', 'dodgerblue', 'aqua',
                       'dimgray', 'dimgray',
                       'mediumaquamarine', 'teal', 'dimgray', 'coral',
                       'dimgray', 'orange', 'purple'])
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'K', 'I', 'J', 'L', 'M', 'N',
              'O', 'P', 'Q', 'R', 'S', 'T', 'U']

    fs = 18
    projection = ccrs.PlateCarree(300)
    sns.set()

    ddeg = 1  # resolution of the binning
    sp = 6  # sinking speed m/day
    season = 'winter' # ''
    dd = 10  # the depth at which the particle backtracking was stopped (10m)
    res = 1  # resolution of the release lcoations

    K = 150  # Number of eigenvectors computed
    L = 150  # Number of clusters
    colors = colors[:L]
    labels = labels[:L]
    tmdir = '/Volumes/HD/network_clustering/PT/Transportation_matrix/'

    # Load the Transportation matrix file
    if(season==''):
        if(type(sp) == str):
            data = np.load(tmdir + 'TMglobal_bin'+str(ddeg)+'_dd'+str(int(dd)) +
                           '_sp'+sp+"_res"+str(res) + '.npz')
        else:
            data = np.load(tmdir + 'TMglobal_bin'+str(ddeg)+'_dd'+str(int(dd)) +
                           '_sp'+str(int(sp))+"_res"+str(res) + '.npz')
    else:
        data = np.load(tmdir + 'TMglobal_bin'+str(ddeg)+'_dd'+str(int(dd)) +
                           '_sp'+str(int(sp))+"_res"+str(res) + 
                           '_season%s'%(season) + '.npz')

    TM = data['TM'][:]
    Lons = data['Lons'][:]
    Lats = data['Lats'][:]
    vLons, vLats = np.meshgrid(Lons, Lats)
    vLons = vLons.flatten()
    vLats = vLats.flatten()
    exte = [1, 360, -75, 75]  # used extend (longitudes and latitudes)
    #%%Create the bipartite graph
    bottom_boxes = [-i for i in range(len(vLons))]
    surface_boxes = range(len(vLons))

    B = nx.Graph()  # create graph
    B.add_nodes_from(surface_boxes,
                     bipartite=0)  # Add the surface boxes as nodes
    B.add_nodes_from(bottom_boxes,
                     bipartite=1)  # Add the bottom boxes as nodes
    B.add_edges_from(nwf.get_edges(TM))  # Add the edges
    #%%
    G = bipartite.weighted_projected_graph(B, bottom_boxes)  # Create bipartite graph
    A = nwf.undirected_network(csr_matrix(nx.adjacency_matrix(G)))
    A1 = A.largest_connected_component()  # Obtain the largest connected component
    w, v = A1.compute_laplacian_spectrum(K=max(40, K))
    A1.hierarchical_clustering_ShiMalik(K)
    networks = A1.clustered_networks
    #%%
    Z = nwf.construct_dendrogram([networks[i] for i in range(L)])

    field_plot = np.ones(nx.adjacency_matrix(G).shape[0]) * (-10000)
    for k in range(L):
        field_plot[networks[L-1][k].cluster_indices] = networks[L-1][k].cluster_label
    field_plot = np.ma.masked_array(field_plot, field_plot == -10000)

    networks0 =  copy(networks)

    netn = []
    for i in list(networks0.keys()):
        netn.append(networks0[i])
    networks = np.array(netn)
    #%% Write file
    print('write files')
    dirWrite = '/Volumes/HD/network_clustering/clusteroutput/'
    if(season==''):
        np.savez(dirWrite+'hier_clus_sp%d_exte%d_%d_%d_%d_K%d_L%d'%(sp, exte[0],
                                                                    exte[1],
                                                                    exte[2],
                                                                    exte[3], K, L),
                dendogram=Z, L=[L], K=[K], labels=labels, w=w, v=v,
                field_plot=field_plot, vLats=vLats, vLons=vLons)
        np.save(dirWrite+'hier_clus_sp%d_exte%d_%d_%d_%d_K%d_L%d'%(sp, exte[0],
                                                                   exte[1],
                                                                   exte[2],
                                                                   exte[3], K, L),
                networks)
    else:
        np.savez(dirWrite+'hier_clus_sp%d_exte%d_%d_%d_%d_K%d_L%d_%s'%(sp, exte[0],
                                                                    exte[1],
                                                                    exte[2],
                                                                    exte[3], K,
                                                                    L, season),
                dendogram=Z, L=[L], K=[K], labels=labels, w=w, v=v,
                field_plot=field_plot, vLats=vLats, vLons=vLons)
        np.save(dirWrite+'hier_clus_sp%d_exte%d_%d_%d_%d_K%d_L%d_%s'%(sp, exte[0],
                                                                   exte[1],
                                                                   exte[2],
                                                                   exte[3], K,
                                                                   L, season),
                networks)
