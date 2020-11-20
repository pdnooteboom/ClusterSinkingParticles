#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 16:08:11 2020

@author: nooteboom
"""

import os
assert os.environ['CONDA_DEFAULT_ENV']=='Cartopy-py3', 'You should use the conda environment Cartopy-py3'
import numpy as np
import matplotlib.pylab as plt
import cartopy.crs as ccrs
import seaborn as sns
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import cartopy.feature as cfeature


if(__name__=='__main__'):
    sns.set(context='paper', style='whitegrid')
    
    sp = 250 # the sinking speed (m/day)
    mins = 300 # The variable s_{min} that is used for subplots (a) and (b)
    fs = 22 # the fontsize for plotting
    si=8 # the markersize for plotting
    # Set the color bounds for the reachability (vmin and vmax)
    if(sp==6):
        vs = [7000,20000]
    elif(sp==25):
        vs = [7000,13000]
    elif(sp==250):
        vs = [7000,14000]
    elif(sp==11):
        vs = [7000,15000]
    markers = 10 # another markersize for plotting
    noisecolor = 'k' # the color for plotting
    #%% Load the OPTICS result
    dirr = ''
    ff = np.load(dirr+'OPTICS_sp%d_smin%d.npz'%(sp, mins))
    lon0 = ff['lon']
    lat0 = ff['lat']
    reachability = ff['reachability']
    ordering = ff['ordering']
    #%%
    exte=[18, 360-70, -75, 75]; latlines=[-75,-50, -25, 0, 25, 50, 75, 100];
    
    f = plt.figure(constrained_layout=True, figsize = (20, 12))
    gs = f.add_gridspec(2, 8)
    
    ax = f.add_subplot(gs[0,3:],projection=ccrs.PlateCarree())
    ax.set_title('(b)', fontsize=fs)
    ax.add_feature(cfeature.LAND, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)
    g = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
    g.xlocator = mticker.FixedLocator([-180,-90, -0, 90, 180])
    g.xlabels_top = False
    g.ylabels_left = False
    g.xlabel_style = {'fontsize': fs}
    g.ylabel_style = {'fontsize': fs}
    g.xformatter = LONGITUDE_FORMATTER
    g.yformatter = LATITUDE_FORMATTER
    g.ylocator = mticker.FixedLocator(latlines)
    
    ax.set_extent(exte, ccrs.PlateCarree())
    p = ax.scatter(lon0, lat0, s=25, c=reachability, cmap = 'inferno',
                              alpha=0.6, vmin=vs[0], vmax=vs[1])
    
    cbar = plt.colorbar(p, shrink=.8, aspect=10, orientation='horizontal', extend='max')
    
    
    ax = f.add_subplot(gs[0,:3])
    
    ax.scatter(np.arange(len(reachability)), 
               reachability[ordering], 
               c=noisecolor, 
                    marker="o", s=5, alpha=0.1)
    ax.set_title('(a)', size=10, fontsize=fs)
    ax.set_ylabel(r"$r(p_i)$", fontsize=fs)
    ax.set_xlabel(r"$i$", fontsize=fs)
    ax.tick_params(labelsize=fs)
    
    #%% Mantel part
    minss = [100, 200, 300, 400, 500, 600,700,800,900, 1000] # s_min values
    perm = 999
    ff = np.load('distance_matrices/mantel_tests_sp%d_perm%d.npz'%(sp,perm))
    
    DR = ff['DR'][:,0] # Dinocyst R-statistics
    DP = ff['DP'][:,0] # Dinocyst p-value
    FR = ff['FR'][:,0] # Foraminifera R-statistics
    FP = ff['FP'][:,0] # Foraminifera p-value
    
    plt.rcParams['axes.axisbelow'] = True
    lw = 3
    si = 10
    
    custom_lines = [Line2D([0], [0],linestyle='--', color='k', lw=lw, marker='o', markersize=si),
                    Line2D([0], [0],linestyle='-', color='k', lw=lw, marker='o', markersize=si)]
    
    ticks1 = [0,0.1,0.2,0.3, 0.4, 0.5]
    
    ax = f.add_subplot(gs[1,2:-2])
    
    color = 'k'
    ax.set_ylabel('correlation \n (partial Mantel)', color=color, fontsize=fs) 
    ax.plot(minss,DR,'--o', color=color, lw=lw, markersize=si)
    ax.plot(minss,FR,'-o', color=color, lw=lw, markersize=si, label='')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_yticks(ticks1)
    ax.set_yticklabels(ticks1, fontsize=fs)
    ax.set_ylim(-0.01,0.52)
    ax.set_xlabel('$s_{min}$', fontsize=fs)
    ax.set_title('(c)', size=10, fontsize=fs)
    for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fs) 
    
    ax.legend(custom_lines, ['Dinocysts','Foraminifera'], fontsize=fs, loc='best')
                    
    #%% save figure
    
    f.savefig("reach_mins%d_Mantel_sp%d"%(mins,sp), dpi=300)
    plt.show()
