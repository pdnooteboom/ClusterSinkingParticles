#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:05:15 2020

@author: nooteboom
"""
import os
assert os.environ['CONDA_DEFAULT_ENV']=='Cartopy-py3', 'You should use the Cartopy-py3 conda environment here'
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import plot_functions as plf
sns.set(style='whitegrid', context='paper', font_scale=2)


spl = [6,11,25,250]#
fs = 20
biot = 'shannon'#'rich'#'CHAO'#'even'#   # The biodiversity index used
extend='glob'#'SO'  # whether to use the global sedimentary data or in the Southern Hemisphere
    
#%% Plot the heatmaps
fs=20
maxxi = 0.1
      
for sp in spl: # for every sinking speed in spl
    print(sp)
    if('extend'=='SO'):
        if('even'==biot):
            ff = np.load('cb_res/cb_even_%d.npz'%(sp))        
        elif('rich'==biot):
            ff = np.load('cb_res/cb_%d.npz'%(sp))
        elif('CHAO'==biot):
            ff = np.load('cb_res/cb_chao_%d.npz'%(sp))
        elif('shannon'==biot):
            ff = np.load('cb_res/cb_shannon_%d.npz'%(sp))
    elif(extend=='glob'):
        if('shannon'==biot):
            ff = np.load('cb_res/cbG_shannon_%d.npz'%(sp))
    minss = ff['smins']
    xiss = ff['xiss']
    Dbiod = ff['Dbiod'][:]
    Fbiod = ff['Fbiod'][:]
    if('even'==biot):
        vm = [-1,1]  
        cmap1 = 'seismic'
        cmap1.set_bad("tab:blue")
    elif('shannon'==biot):
        if(extend=='SO'):
            vm = [-2.5,2.5]  
        elif(extend=='glob'):
            vm = [-0.7,0.7]  
        cmap1 = plf.balance_colormap2(vm)
        cmap1.set_bad("tab:blue")
    else:
        vm = [min(np.nanmin(Dbiod),np.nanmin(Fbiod)-0.1, -1),15]
        cmap1 = plf.balance_colormap(vm)
        cmap1.set_bad("tab:blue")
    Dbiod[Dbiod==0] = np.nan
    Fbiod[Fbiod==0] = np.nan
    # Set the x ticks:
    num_ticks = len(xiss)
    frac = 9
    xticks = np.linspace(0, (len(xiss) - 1), num_ticks, dtype=np.int)
    xticklabels = []
    for i, idx in enumerate(xticks):
        if(i%frac==0):
            xticklabels.append(np.round(xiss[idx],5))
        else:
            xticklabels.append('')

    DN = pd.DataFrame(data=Dbiod, index=minss, columns=xiss)
    FN = pd.DataFrame(data=Fbiod, index=minss, columns=xiss)

    # The figure
    fig, ax = plt.subplots(1,3, figsize=(16,8),
                           gridspec_kw={'width_ratios':[1,1,0.08]})
    ax[0].get_shared_y_axes().join(ax[1])
    g1 = sns.heatmap(DN,cmap=cmap1,cbar=False,ax=ax[0], vmin=vm[0], vmax=vm[1], 
                     xticklabels=xticklabels, 
                     cbar_kws={'label': '$\overline{N}_c - \overline{N}_{nc}$'})
    ax[0].set_xticklabels(xticklabels, fontsize=fs-6, rotation='vertical')
    ax[0].set_xticks(xticks+0.5)
    ax[0].set_yticklabels(minss,fontsize=fs-6, rotation='horizontal')
    g1.set_ylabel('$s_{min}$', fontsize=fs)
    g1.set_title('(a) dinocysts', fontsize=fs)
    g1.set_xlabel('$\\xi$', fontsize=fs)
    g2 = sns.heatmap(FN,cmap=cmap1,cbar=True,ax=ax[1], vmin=vm[0], vmax=vm[1], 
                     cbar_ax=ax[2], yticklabels=False)
    g2.collections[0].colorbar.set_label('$\overline{N}^{nc}_s - \overline{N}^{c}_s$', 
                  fontsize=fs)
    g2.collections[0].colorbar.extend = 'max'
    ax[1].set_xticklabels(xticklabels, fontsize=fs-6, rotation='vertical')
    ax[1].set_xticks(xticks+0.5)
    g2.set_title('(b) foraminifera', fontsize=fs)
    g2.set_xlabel('$\\xi$', fontsize=fs)
    
    if(extend=='SO'):
        if('even'==biot):
            plt.savefig('heatmap_biod_even_sp%d.pdf'%(sp), dpi=300,bbox_inches='tight')
        elif('CHAO'==biot):
            plt.savefig('heatmap_biod_chao_sp%d.pdf'%(sp), dpi=300,bbox_inches='tight')
        elif('shannon'==biot):
            plt.savefig('heatmap_biod_shannon_sp%d.png'%(sp), dpi=300,bbox_inches='tight')
        else:
            plt.savefig('heatmap_biod_rich_sp%d.pdf'%(sp), dpi=300,bbox_inches='tight')
    elif(extend=='glob'):
        if('shannon'==biot):
            plt.savefig('heatmapG_biod_shannon_sp%d.png'%(sp), dpi=300,bbox_inches='tight')
    plt.show()