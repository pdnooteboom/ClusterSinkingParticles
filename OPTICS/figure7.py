#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:05:15 2020

@author: nooteboom
"""
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import matplotlib.colors as mcolors
import pandas as pd
sns.set(style='whitegrid', context='paper', font_scale=2)
import os

assert os.environ['CONDA_DEFAULT_ENV']=='Cartopy-py3', 'You should use the Cartopy-py3 conda environment here'


spl = [6,11,25,250]#
fs = 20
biot = 'shannon'#'rich'#'CHAO'#'even'#
extend='glob'#'SO'

#minss = np.array([50,200,300,400,500,600,700, 800, 900, 1000])
#for sp in spl:
#    print(sp)
#    ff = np.load('cb_res/cbd_%d.npz'%(sp), allow_pickle=True)
#    xiss = ff['xiss']
##    minss = ff['smins']
#    Dbiod = ff['Dbiod']
#    Fbiod = ff['Fbiod']
#            
#    fig, ax = plt.subplots(1,2, sharey=True, sharex=True, figsize=(10,8))
#    ax[0].set_xlim(min(xiss), max(xiss))
#    ax[0].set_xscale('log')
#    for mini,mins in enumerate(minss):
#    #    ax[0].scatter(xiss, Dbiod[mini])
#        ax[0].plot(xiss, Dbiod[mins], '-o', label=mins)
#    ax[0].set_title('Dinocysts')
#    ax[0].set_xlabel('$\\xi$', fontsize=fs)
#    ax[0].set_ylabel('$\overline{N}_c / \overline{N}_{nc}$', fontsize=fs)
#    
#    for mini,mins in enumerate(minss):
#    #    ax[1].scatter(xiss, Fbiod[mini])
#        ax[1].plot(xiss, Fbiod[mins], '-o', label=mins)
#    ax[1].set_title('Foraminifera')
#    ax[1].legend(loc='right',bbox_to_anchor=(1, 0.2, 0.5, 0.5), fontsize=fs,
#      title='$s_{min}$')
#    ax[1].set_xlabel('$\\xi$', fontsize=fs)
#    plt.savefig("richness_sp%d"%(sp), dpi=300)
#    plt.show()
#   
#    
#    
#for sp in spl:
#    print(sp)
#    ff = np.load('cb_res/cb_%d.npz'%(sp))
#    xiss = ff['xiss']
##    minss = ff['smins']
#    Dbiod = ff['Dbiod'][1:]
#    Fbiod = ff['Fbiod'][1:]
#            
#    fig, ax = plt.subplots(1,2, sharey=True, sharex=True, figsize=(16,8))
#    ax[0].set_xlim(min(xiss), max(xiss))
#    ax[0].set_xscale('log')
#    for mini,mins in enumerate(minss):
#    #    ax[0].scatter(xiss, Dbiod[mini])
#        ax[0].plot(xiss, Dbiod[mini], '-o', label=mins)
#    ax[0].set_title('Dinocysts', fontsize=fs)
#    ax[0].set_xlabel('$\\xi$', fontsize=fs)
#    ax[0].set_ylabel('$\overline{N}_c / \overline{N}_{nc}$', fontsize=fs)
#    
#    for mini,mins in enumerate(minss):
#    #    ax[1].scatter(xiss, Fbiod[mini])
#        ax[1].plot(xiss, Fbiod[mini], '-o', label=mins)
#    ax[1].set_title('Foraminifera', fontsize=fs)
#    legend = ax[1].legend(loc='right',bbox_to_anchor=(1, 0.2, 0.5, 0.5), fontsize=fs,
#      title='$s_{min}$')
#    ax[1].set_xlabel('$\\xi$', fontsize=fs)
#    plt.savefig("richness_sp%d"%(sp), dpi=300, bbox_inches='tight')
#    plt.setp(legend.get_title(),fontsize=fs)
#    plt.show()
    
#%% And heatmaps
import seaborn as sns
sns.set(style='whitegrid',context='paper', font_scale=2)
fs=20
maxxi = 0.1

def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()

def balance_colormap(vm):
    lb = int(vm[0])
    ub = int(vm[1])
    print(lb, ub)
    mp=200
#    colors1 = plt.cm.Blues(np.linspace(0., 1, 128))[-50:-48]
    
    colors2 = plt.cm.RdGy(np.linspace(0, 1, lb*-mp))[lb*-mp//2+1:][::-1]
    colors3 = plt.cm.PRGn(np.linspace(0, 1, ub*mp))[-ub*mp//2+140:]

#    colors1 = plt.cm.Blues(np.linspace(0., 1, 128))[-50:-48]
#    colors2 = plt.cm.RdGy(np.linspace(0, 1, 1000))[500:][::-1][-lb*50:]
#    colors3 = plt.cm.PRGn(np.linspace(0, 1, 1000))[-500:][:ub*50]
    # combine them and build a new colormap
    colors = np.vstack(( colors2, colors3))#
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    return cmap

def balance_colormap2(vm):
#    colors1 = plt.cm.Blues(np.linspace(0., 1, 128))[-50:-48]
    
#    colors2 = plt.cm.RdGy(np.linspace(0, 1, 66))[33:][::-1]
#    colors3 = plt.cm.PRGn(np.linspace(0, 1, 135))[66:]
    colors2 = plt.cm.RdGy(np.linspace(0, 1, 100))[50:][::-1]
    colors3 = plt.cm.PRGn(np.linspace(0, 1, 100))[50:]
    
    # combine them and build a new colormap
    colors = np.vstack((colors2, colors3))#
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    return cmap

# Create the colormap:                
#colors1 = plt.cm.Blues(np.linspace(0., 1, 128))[-50:-47]
#colors3 = plt.cm.RdGy(np.linspace(0, 1, 100))[50:]
#colors2 = plt.cm.PRGn(np.linspace(0, 1, 100))[::-1][:50]


#minss = np.array([50,200,300,400,500,600,700, 800, 900, 1000])
        
for sp in spl:
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
        cmap1 = balance_colormap2(vm)
        cmap1.set_bad("tab:blue")
    else:
        vm = [min(np.nanmin(Dbiod),np.nanmin(Fbiod)-0.1, -1),15]
        cmap1 = balance_colormap(vm)
        cmap1.set_bad("tab:blue")
#    Dbiod[np.isnan(Dbiod)] = -200
#    Fbiod[np.isnan(Fbiod)] = -200
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
#    g2.collections[0].colorbar.set_label('$\overline{N}_c / \overline{N}_{nc}$', fontsize=fs)
    g2.collections[0].colorbar.set_label('$\overline{N}^{nc}_s - \overline{N}^{c}_s$', fontsize=fs)
    g2.collections[0].colorbar.extend = 'max'
    ax[1].set_xticklabels(xticklabels, fontsize=fs-6, rotation='vertical')
    ax[1].set_xticks(xticks+0.5)
    #g2.set_ylabel('$s_{min}$')
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