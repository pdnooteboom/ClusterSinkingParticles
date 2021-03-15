#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:49:01 2020

@author: nooteboom
"""
import os
assert os.environ['CONDA_DEFAULT_ENV'] in ['Cartopy-py3','Cartopy-py32'], 'You should use the Cartopy_py3 conda environment here'
import numpy as np
from sklearn.cluster import  cluster_optics_xi, cluster_optics_dbscan
import matplotlib
import matplotlib.pylab as plt
import network_functions as nwf
import cartopy.crs as ccrs
import seaborn as sns
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
from scipy.spatial.distance import pdist, squareform
from matplotlib.lines import Line2D

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

def get_colors(sp=6):
    # create the colors that are used for plotting
    colo = ["gist_ncar","Reds"]#Greys"]
    colorsg = []
    if(sp==6):
        its = 13
    else:
        its = 15
    colorsg.append(sns.color_palette(colo[1], n_colors=its+1)[4])
    for k in range(its):
        colorsg.append(sns.color_palette(colo[0], n_colors=its+1)[k])
    colorsg.reverse()
    return colorsg

def shannon(vec):
    # shannon biodiversity
    assert len(vec.shape)==1, 'vec should be a vector'
    vec = vec / np.sum(vec)
    h = 0
    for i in range(len(vec)):
        if(vec[i]!=0):
            h += -1*vec[i]*np.log(vec[i])
    return h

    #%%

if(__name__=='__main__'):        
    sns.set(context='paper', style='whitegrid')
    sp = 6 # the sinking speed (m/day)
    colorsg = get_colors(sp)
    mins = 300 # The s_min parameter
    alpha = 1 # alpha for plotting points
    maxlat = 75 # maximum latitude considered for clustering
    maxlatS = 5 # maximum latitude considered for sedimentary microplankton data
    # Set the xi parameter that is used for clustering
    if(sp==6):
        if(mins==300):
            opts = [
                    ["xi", 0.002] # use xi parameter
                    ]
    elif(sp==11):
        if(mins==200):
            opts = [
                    ["xi", 0.004]
                    ]
    markers = 12 # markersize for plotting

    fs=25 # fontsize for plotting
    noisecolor = 'gray' # color for plotting of noisy points
    #%% Load the OPTICS result
    dirr = '/Users/nooteboom/Documents/GitHub/cluster_TM/cluster_SP/density/dens/results/'
    ff = np.load(dirr+'OPTICS_sp%d_smin%d.npz'%(sp, mins))
    lon0 = ff['lon']
    lat0 = ff['lat']
    reachability = ff['reachability'] / 1000
    ordering = ff['ordering']
    predecessor = ff['predecessor']
    core_distances = ff['core_distances']
    #%% Create the clusters from the reachabilities, given the xi value
    labels = []
    for op in opts:
        m, c = op[0], op[1]
        if m == "xi":
            l, _ = cluster_optics_xi(reachability, predecessor, ordering, mins, xi=c)
        else:
            l = cluster_optics_dbscan(reachability=reachability,
                                                core_distances=core_distances,
                                               ordering=ordering, eps=c)
        labels.append(l) 
    
    norms = []
    for l in labels:
        bounds = np.arange(-.5,np.max(l)+1.5,1)
        norms.append(matplotlib.colors.BoundaryNorm(bounds, len(bounds)))
    
    #%%
    exte=[18, 360-70, -75, 0]; latlines=[-75,-50, -25, 0, 25, 50, 75, 100];
    
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
    Dinodata[np.isnan(Dinodata)] = 0
    idxgz = (Dinodata.sum(axis=1)!=0)
    FlonsDino[FlonsDino<0] += 360
    FlonsDino = FlonsDino[idxgz]
    FlatsDino = FlatsDino[idxgz]
    Dinodata = Dinodata[idxgz]
    
    # use only sedimentary data below maxlatS degrees N
    idx = np.where(Flats<maxlatS)[0]
    Flons = Flons[idx]
    Flats = Flats[idx]
    data = data[idx]
    idx = np.where(FlatsDino<maxlatS)[0]
    FlonsDino = FlonsDino[idx]
    FlatsDino = FlatsDino[idx]
    Dinodata = Dinodata[idx]
    
    # apply the multidimensional scaling to the microplankton data
    Fvalue, XF = MDS_data(data)
    Dinovalue, XDino = MDS_data(Dinodata)    
    #%%
    #Optics results
    f = plt.figure(constrained_layout=True, figsize = (20, 13))
    gs = f.add_gridspec(2, 10)
    
    gs_s = [[[0,1],[0,2]], [[1,1], [1,2]], [[2,1], [2,2]], [[3,1],[3,2]], [[4,1],[4,2]]]
    panel_labels = [['(a) ','(b) ','(c) '],['(d) ', '(e) ','(f) '],['(g) ', '(h) ','(i) '],['(j) ', '(k) ','(l) ']]
    
    labels = np.array(labels)[0]
    labels[lat0>maxlat] = -1
    
    
    args = nwf.find_nearest_args(lon0, lat0, Flats, Flons)
    reachdata = reachability[args]
    Flabels = labels[args]
    Dinoargs = nwf.find_nearest_args(lon0, lat0, FlatsDino, FlonsDino)
    reachDinodata = reachability[Dinoargs]
    Dinolabels=labels[Dinoargs]
    
    idx = (Flabels!=-1)
    idx0 = (Flabels==-1)
    xF1 = XF[:,0][idx]
    xF2 = XF[:,1][idx]
    labelsF = Flabels[idx]
    xF01 = XF[:,0][idx0]
    xF02 = XF[:,1][idx0]
    
    idx = (Dinolabels!=-1)
    idx0 = (Dinolabels==-1)
    xD1 = XDino[:,0][idx]
    xD2 = XDino[:,1][idx]
    labelsD = Dinolabels[idx]
    xD01 = XDino[:,0][idx0]
    xD02 = XDino[:,1][idx0]
    
    tot_clus = np.unique(labels)
    
    #Reachability plot
    i = 1
    Labels_clusters = np.ma.masked_where(labels <0, labels, copy=True)
    Labels_noise = [noisecolor] * len(labels)
    Labels_noise =  np.ma.masked_where(labels >= 0, Labels_noise, copy=True)

    ax = f.add_subplot(gs[0,6:])
    w0 =  np.where(~Labels_noise[ordering].mask)
    reachnoise = reachability[ordering][w0]
    for li,l in enumerate(tot_clus):
        w0 = np.where(labels[ordering]==l)
        if(l==-1):        
            ax.scatter(np.arange(len(reachability))[w0], 
                       reachnoise, 
                       c=noisecolor, 
                            marker="o", s=5, alpha=0.1)
        else:
            ax.scatter(np.arange(len(reachability))[w0], reachability[ordering][w0], 
                       c=colorsg[li], marker="o", s=5)


    if opts[0][0] == "xi":
        a, b = r"$\xi$", opts[0][1]
    else:
        a, b = r"$\epsilon$", opts[0][1]
        ax.axhline(opts[0][1], color="k")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.set_title(panel_labels[0][1] +  a + ' = ' + str(b), size=10, fontsize=fs)
    ax.set_ylabel(r"$10^{-3}\cdot r(p_i)$", fontsize=fs)

    ax.set_xlabel(r"Sediment location # $i$", fontsize=fs)
    ax.tick_params(labelsize=fs)

    ax = f.add_subplot(gs[0,:6],projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, zorder=0, color=noisecolor)
    g = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
    g.xlocator = mticker.FixedLocator([-180,-90, -0, 90, 181])
    g.xlabels_top = False
    g.ylabels_right = False
    g.xlabel_style = {'fontsize': fs}
    g.ylabel_style = {'fontsize': fs}
    g.xformatter = LONGITUDE_FORMATTER
    g.yformatter = LATITUDE_FORMATTER
    g.ylocator = mticker.FixedLocator(latlines)
    
    ax.set_extent(exte, ccrs.PlateCarree())

    for li,l in enumerate(tot_clus):
        w0 = np.where(labels==l)
        if(l!=-1):  
            p = ax.scatter(lon0[w0], lat0[w0], s=markers, c=colorsg[li],
                              alpha=alpha, zorder=9) 
    ax.set_title(panel_labels[0][0], size=10, fontsize=fs)
    ax.add_feature(cfeature.LAND, zorder=100, color='peachpuff')
    ax.add_feature(cfeature.COASTLINE, zorder=100)
    ax.set_ylim(-75,maxlat)

    # Add a scatter of the sediment sample sites
    sc = ax.scatter(Flons, Flats, s=20, marker='X',
               zorder=10)
    sc.set_facecolor("k")
    sc = ax.scatter(FlonsDino, FlatsDino, s=30, marker='o',
               zorder=10000, edgecolor='k')
    sc.set_facecolor("white")
    
    custom_lines = [Line2D([0], [0], marker='o', markerfacecolor='white', 
                           markeredgecolor='k', 
                           lw=0, markersize=9),
                    Line2D([0], [0], marker='X', markerfacecolor='k', 
                           markeredgecolor='k',
                           lw=0, markersize=9)]
    
    legend = ax.legend(custom_lines, ['dinocyst', 
                                      'foraminifera'], 
                       bbox_to_anchor=(1., 1.37), loc='upper right', ncol=1,
                       facecolor='darkgrey', fontsize=fs-3,
                       title='sample sites', title_fontsize=fs-2)

#%%The MDS part    
    print('taxonomical distance versus clusters')
    ax = f.add_subplot(gs[1,5:])
    
    for li,l in enumerate(tot_clus):
        w0 = np.where(labelsF==l)
        if(l==-1):  
            ax.scatter(xF01, xF02, c=noisecolor, s=10,alpha=0.5, marker='X')
        else:
            ax.scatter(xF1[w0], xF2[w0], c=colorsg[li], s=80, 
                       alpha=alpha, marker='X')
    
    ax.set_title('(d) Foraminifera', fontsize=fs)
    ax.set_xlabel('first MDS axis', fontsize=fs)
    ax.set_ylabel('second MDS axis', fontsize=fs)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fs)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)


    ax = f.add_subplot(gs[1,:5])   
    
    for li,l in enumerate(tot_clus):
        w0 = np.where(labelsD==l)
        if(l==-1):  
            ax.scatter(xD01, xD02, c=noisecolor, s=25,alpha=0.5,marker='o')
        else:
            p = ax.scatter(xD1[w0], xD2[w0], c=colorsg[li], s=50, 
                           alpha=alpha,marker='o')
    
    ax.set_title('(c) Dinocysts', fontsize=fs)
    ax.set_xlabel('first MDS axis', fontsize=fs)
    ax.set_ylabel('second MDS axis', fontsize=fs)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fs)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)
        
    #%% Save the figure
    f.savefig("figs/optics_sp%d_mins%d_withMDS"%(sp,mins), dpi=600, bbox_inches='tight')
    plt.show()
        
    #%% print the mean biodiversity including and excluding noisy locations
    if(False):
        meanSn = 0.
        meanSc = 0.
        co = 0
        for i in range(data.shape[0]):
            meanSn += shannon(data[i])
            if(Flabels[i]!=-1):
                co += 1
                meanSc += shannon(data[i])
          
        print('mean Foram biodiversity including and excluding noisy locations:')
        print(meanSn/data.shape[0],'   ',meanSc/co)
    
        meanSn = 0.
        meanSc = 0.
        co = 0
        for i in range(Dinodata.shape[0]):
            meanSn += shannon(Dinodata[i])
            if(Dinolabels[i]!=-1):
                co += 1
                meanSc += shannon(Dinodata[i])
          
        print('mean Dino biodiversity including and excluding noisy locations:')
        print(meanSn/Dinodata.shape[0],'   ',meanSc/co)
