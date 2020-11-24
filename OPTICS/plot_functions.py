#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:51:22 2020

@author: nooteboom
"""

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pylab as plt

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
    mp=200
    
    colors2 = plt.cm.RdGy(np.linspace(0, 1, lb*-mp))[lb*-mp//2+1:][::-1]
    colors3 = plt.cm.PRGn(np.linspace(0, 1, ub*mp))[-ub*mp//2+140:]

    # combine them and build a new colormap
    colors = np.vstack(( colors2, colors3))#
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    return cmap

def balance_colormap2(vm):
    colors2 = plt.cm.RdGy(np.linspace(0, 1, 100))[50:][::-1]
    colors3 = plt.cm.PRGn(np.linspace(0, 1, 100))[50:]
    
    # combine them and build a new colormap
    colors = np.vstack((colors2, colors3))#
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    return cmap