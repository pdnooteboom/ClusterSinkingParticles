#!/usr/bin/env python3
"""
Script with functions that are used for plotting figure 2
"""
# -*- coding: utf-8 -*-
import matplotlib.pylab as plt
import cartopy.crs as ccrs
import seaborn as sns
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
import matplotlib
import numpy as np
from matplotlib.lines import Line2D

def plot_bounds(lo, la, direc, c='k', linewidth=2, ax=None):
    plo = []
    pla = []
    for l in range(len(lo)):
        if(direc[l]==0):
            plo.append([lo[l],lo[l]])
            pla.append([la[l]-0.5,la[l]+0.5])
        else:
            plo.append([lo[l]-0.5,lo[l]+0.5])
            pla.append([la[l],la[l]])
    if(ax==None):
        for l in range(len(plo)):
            plt.plot(plo[l], pla[l], c=c, zorder=9, linewidth=linewidth)
    else:
        for l in range(len(plo)):
            ax.plot(plo[l], pla[l], c=c, zorder=9, linewidth=linewidth)

def plot_bounds_nf(lo, la, direc,flo, fla, fdirec, c='k', linewidth=2):
    plo = []
    pla = []
    for l in range(len(lo)):
        if(direc[l]==0):
            plo.append([lo[l],lo[l]])
            pla.append([la[l]-0.5,la[l]+0.5])
        else:
            plo.append([lo[l]-0.5,lo[l]+0.5])
            pla.append([la[l],la[l]])
    for l in range(len(plo)):
        plt.plot(plo[l], pla[l], c=c, zorder=9, linewidth=linewidth)

def geo_Fdata2(lats, lons, coms, ax=plt.axes(projection=ccrs.PlateCarree()),
               projection=ccrs.PlateCarree(300), fs=16, 
              exte=[18, 360-70, -75, 0], latlines=[-75,-50, -25, 0, 25, 50, 75, 100] ,
              cmap=None, norm=None, label=None, ncol=1, bboxan = (0., -0.07),
               title='Clusters'):
    ax.add_feature(cfeature.LAND, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10) 
    ax.set_title(title, fontsize=fs)
    
    g = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    g.xlocator = mticker.FixedLocator([-90, -0, 90, 180])
    g.xlabels_top = False
    g.ylabels_right = False
#    g.xlabels_bottom = False
    g.xlabel_style = {'fontsize': fs}
    g.ylabel_style = {'fontsize': fs}
    g.xformatter = LONGITUDE_FORMATTER
    g.yformatter = LATITUDE_FORMATTER
    g.ylocator = mticker.FixedLocator(latlines)
    
    ax.set_extent(exte, ccrs.PlateCarree())
    idx = ~coms.mask
    plotdata = {}; plotdata['lons'] = lons[idx]; plotdata['lats'] =lats[idx];
    plotdata['coms'] = np.array([label[int(j)] for j in coms[idx]])
    sns.scatterplot(data=plotdata, x='lons', y='lats', hue='coms', palette=cmap, 
                    norm=norm, linewidth=0, zorder=9)
    ax.get_legend().remove()
    
def geo_Fdata_bounds(lats, lons, directions, ax = plt.axes(projection=ccrs.PlateCarree()), 
                     projection=ccrs.PlateCarree(0), fs=16, 
              exte=[18, 360-70, -75, 75], latlines=[-75,-50, -25, 0, 25, 50, 75, 100] ,
              cmap=None, norm=None, label=None,lw=2, title='Hierachy'):
    oceancolor = 'dimgray'
    #landcolor = 'darkgoldenrod'
    
    ax.add_feature(cfeature.NaturalEarthFeature('physical',
                                               'ocean', '50m', 
                                                edgecolor=oceancolor, 
                                                facecolor=oceancolor))

    ax.set_title(title, fontsize=fs)
    
    g = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    g.xlocator = mticker.FixedLocator([-180,-90, -0, 90, 180])
    g.xlabels_top = False
    g.ylabels_right = False
    g.xlabel_style = {'fontsize': fs}
    g.ylabel_style = {'fontsize': fs}
    g.xformatter = LONGITUDE_FORMATTER
    g.yformatter = LATITUDE_FORMATTER
    g.ylocator = mticker.FixedLocator(latlines)
    ax.set_extent(exte, ccrs.PlateCarree())
    
    
    for l in range(len(lons)):
        if(l==len(lons)):
            lon = lons[l]
            lon[lon>180] -= 360
            plot_bounds(lons[l], lats[l], directions[l], c=cmap(l/len(lons)), linewidth=lw)#oceancolor, linewidth=lw)
        elif(l<len(lons)-1):
            lon = lons[l]
            lon[lon>180] -= 360
            plot_bounds(lons[l], lats[l], directions[l], c=cmap(l/len(lons)), linewidth=lw)

    ax.add_feature(cfeature.LAND, zorder=10)#, color=landcolor)
    ax.add_feature(cfeature.COASTLINE, zorder=10) 
    
def plot_colorbar(ax, cmap, vmin=1, vmax=2, label='Iteration'):
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label(label)
    
def geo_Fdata_bounds_black(lats, lons, directions, ax = plt.axes(projection=ccrs.PlateCarree()), 
                           fs=16, lws = [1,2]):
    
    for l in range(len(lons)):
        lwi = min((l // (len(lons)//len(lws)) ), len(lws)-1)
        if(l==len(lons)):
            lon = lons[l]
            lon[lon>180] -= 360
            plot_bounds(lon-0.5, lats[l], directions[l], c='k', 
                        linewidth=lws[0], ax=ax)
        elif(l<len(lons)):
            lon = lons[l]
            lon[lon>180] -= 360
            plot_bounds(lon-0.5, lats[l], directions[l], c='k', 
                        linewidth=lws[lwi], ax=ax)

def plot_legend(ax, its, lws, label='Iteration', fs=15):
    ax.axis('off')
    custom_lines = []
    leg = []
    lws.reverse()
    for lwi in range(len(lws)):
        custom_lines.append(Line2D([0], [0], color='k', lw=lws[lwi]))
        if(lwi<len(lws)-1):
            leg.append(str(lwi*(its//len(lws))) + '-' + str((lwi+1)*(its//(len(lws)))))
        else:
            leg.append(str(lwi*(its//len(lws))) + '-' + str(its))
    legend = ax.legend(custom_lines, leg, title=label, bbox_to_anchor=(.5, 1), 
              loc='upper left', fontsize=fs)
    
    plt.setp(legend.get_title(),fontsize=fs)