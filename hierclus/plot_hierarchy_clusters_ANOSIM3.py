#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:05:10 2020

@author: nooteboom
"""
import os
assert os.environ['CONDA_DEFAULT_ENV']=='Cartopy-py3', 'You should use the conda environment Cartopy-py3'
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import Plotting_functions as plf
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
plt.rcParams['axes.axisbelow'] = True

def get_cmaps():
    co = plt.cm.Wistia(np.linspace(0, 1, 90))[:5]
    co2 = plt.cm.Wistia(np.linspace(0, 1, 90))[10:15]
    co3 = plt.cm.Oranges(np.linspace(0, 1, 90))[20:25]
    co4 = plt.cm.Oranges(np.linspace(0, 1, 90))[30:35]
    co5 = plt.cm.Reds(np.linspace(0, 1, 90))[40:45]
    co6 = plt.cm.Reds(np.linspace(0, 1, 90))[50:55]
    co7 = plt.cm.Blues(np.linspace(0, 1, 90))[60:65]
    co8 = plt.cm.Blues(np.linspace(0, 1, 90))[70:75]
    co9 = plt.cm.Purples(np.linspace(0, 1, 90))[80:85]
    co10 = plt.cm.Greys(np.linspace(0, 1, 100))[95:100]
    colors = np.vstack((co, co2, co3, co4, co5, co6, co7, co8, co9, co10))
    colors2 = np.vstack((co10[::-1], co9[::-1], co8[::-1], co7[::-1], co6[::-1], 
                         co5[::-1], co4[::-1], co3[::-1], co2[::-1], co[::-1]))
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap',colors)
    cmap_r = mcolors.LinearSegmentedColormap.from_list('my_colormap',colors2)
    
    return cmap, cmap_r    

if(__name__=='__main__'):
    its = 90 # number of iterations of hierarchical clustering used for plotting
    sp = 6 # sinking speed (m/day)
    season = 'winter'
    if(sp==6 and season==''):
        K = 600
    else:
        K = 150
    exte = [1, 360, -75, 75]
    
    projection=ccrs.PlateCarree(0)
    lw = 3 # linewidth for plotting
    fs = 18 # fontsize for plotting
    #%% Plot the clusters (for subfig a)
    # load data
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
    # The colors of clusters
    colo = ["Oranges","Blues","Greys","Purples","Greens","Reds"]
    colorsg = sns.color_palette(colo[0], n_colors=18)[1:][:1]
    for k in range(its):
        colorsg += sns.color_palette(colo[(k+1)%len(colo)], n_colors=18)[1:][(-its//len(colo)):(k//len(colo))+1]
    colorsg.reverse()
    labelsg = np.arange(its).astype(str).tolist()
    field_plot = np.ones(dat['field_plot'].shape)*(-10000)
    colors = colorsg[:its]; labels=labelsg[:its];
    bounds = np.arange(-0.5,its+0.5,1)
    norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds))
    cmap0 = matplotlib.colors.ListedColormap(colors)  
    
    for k in range(its): field_plot[networks[its-1][k].cluster_indices]= networks[its-1][k].cluster_label
    field_plot = np.ma.masked_array(field_plot, field_plot==-10000)   
    
    #%%
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 16)
    ax1 = fig.add_subplot(gs[0,:-1], projection=projection)
    
    # subfig a    
    #%
    cmap, cmap_r = get_cmaps() # create colormap for hierarchical bounds
    # Load the boundaries of hierarchical clusters:
    fbounds = np.load('res/cluster%s_bounds_%dits_sp%d.npz'%(season,K,sp),
                      allow_pickle=True)
    lats = fbounds['lats'][::-1][(K-its):-1]
    lons = fbounds['lons'][::-1][(K-its):-1]
    directions = fbounds['directions'][::-1][(K-its):-1]
       
#    ax1 = fig.add_subplot(gs[1,:-1], projection=projection)
    # plot subfig b

    plf.geo_Fdata_bounds(lats, lons, directions, ax=ax1, cmap=cmap_r,
                         projection=projection, lw=lw, fs=18, title='(a)')
    
    # add the colorbar
    ax2 = fig.add_subplot(gs[0,-1])
    plf.plot_colorbar(ax2, cmap, vmax=len(lons))
    
    #%% subfig c, the ANOSIM results
    perm = 999 # amount of permutations
    # Load result
    ff = np.load('Distance_Matrix/ANOSIM_hierarchicalclus%s_sp%d_perm%d_its%d_mlat65.npz'%(season,sp, perm, its))
    FP = ff['ForamP']
    FR = ff['ForamR']
    DinoP = ff['DinoP']
    DinoR = ff['DinoR']
    iterations = np.arange(len(FP))
    
    lw = 2.5 # linewidth
    
    # For the legend
    custom_lines = [Line2D([0], [0],linestyle='--', color='k', lw=lw),
                    Line2D([0], [0],linestyle='-', color='k', lw=lw)]
    
    ticks1 = [-0.1, 0,0.1,0.2,0.3, 0.4, 0.5]
    
    ax = fig.add_subplot(gs[1,1:-2])
    ax.set_axisbelow(True)
    
    color = 'tab:red'
    ax.set_ylabel('ANOSIM \n test-statistic', color=color, fontsize=fs) 
    ax.plot(iterations,DinoR,'--', color=color, lw=lw)
    ax.plot(iterations,FR,'-', color=color, lw=lw, label='')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_yticks(ticks1)
    ax.set_yticklabels(ticks1, fontsize=fs)
    ax.set_ylim(-0.15,0.58)
    ax.set_xlabel('iteration', fontsize=fs)
    for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fs) 
    
    ax.legend(custom_lines, ['Dinocysts','Foraminifera'], fontsize=fs, loc='lower right')

    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(0.0009,1.05)
    ax2.set_axisbelow(True)
    
    color = 'tab:blue'
    ax2.set_title('(b)', fontsize=fs)
    ax2.set_ylabel('p-value', color=color, fontsize=fs) 
    ax2.tick_params(axis='y', labelcolor=color)
#    ticks2 = [0,0.2,0.4,0.6, 0.8]
    ticks2 = [0.01, 0.1, 1]
#    ax2.set_ylim(-0.02,1.04)
    ax2.set_yticks(ticks2)
    ax2.set_yticklabels(ticks2, fontsize=fs)
    ax2.plot(iterations,FP,'-', color=color, lw=lw)
    ax2.plot(iterations,DinoP,'--', color=color, lw=lw)
    
    #%
    plt.savefig('Distance_Matrix/hierarchical_clustering%s_ANOSIM_its%d_sp%d.pdf'%(season,
                                                                                   its,sp),
                bbox_inches='tight', 
                dpi=300)
    plt.show()
