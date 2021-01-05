#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:29:38 2020

@author: nooteboom
"""
import os
assert os.environ['CONDA_DEFAULT_ENV']=='Cartopy-py3', 'You should use the Cartopy_py3 conda environment here'
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
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d

def shannon(vec):
    # calculates the shannon entropy based on a vector of relative abundances
    assert len(vec.shape)==1
    vec = vec / np.sum(vec)
    h = 0
    for i in range(len(vec)):
        if(vec[i]!=0):
            h += -1*vec[i]*np.log(vec[i])
    return h

def get_colors():
    # create the colors that are used for plotting
    colo = ["gist_ncar","Greys"]
    colorsg = []
    its = 13
    colorsg.append(sns.color_palette(colo[1], n_colors=its+1)[-1])
    for k in range(its):
        colorsg.append(sns.color_palette(colo[0], n_colors=its+1)[k])
    colorsg.reverse()
    return colorsg
    #%%

if(__name__=='__main__'):        
    sns.set(context='paper', style='whitegrid')
    colorsg = get_colors()
    sp = 6 # the sinking speed (m/day)
    mins = 300 # The s_min parameter
    alpha = 1 # alpha for plotting points
    maxlat = 75 # maximum latitude considered for clustering
    maxlatS = 5 # maximum latitude considered for sedimentary microplankton data
    # Set the xi parameter that is used for clustering
    if(sp==6):
        if(mins==300):
            opts = [
                    ["xi", 0.002] # use xi clustering
                    ]
    markers = 12 # markersize for plotting

    fs=25 # fontsize for plotting
    noisecolor = 'dimgray' # color for plotting of noisy points
    #%% Load the OPTICS result
    dirr = '/Users/nooteboom/Documents/GitHub/cluster_TM/cluster_SP/density/dens/results/'
    ff = np.load(dirr+'OPTICS_sp%d_smin%d.npz'%(sp, mins))
    lon0 = ff['lon']
    lat0 = ff['lat']
    reachability = ff['reachability']
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
    exte=[18, 360-70, -75, 0]; latlines=[-70,-60,-50, -40, 0, 25, 50, 75, 100];
    
    readData = '/Volumes/HD/network_clustering/'
    
    # Read Dino data
    FlonsDino, FlatsDino, Dinodata = nwf.readDinoset(readData+'dinodata_red.csv')
    stations = nwf.readDinoset_stations(readData+'dinodata_stations.csv')
    species = nwf.readDinoset_species(readData+'dinodata_species.csv')
    Dinodata[np.isnan(Dinodata)] = 0
    idxgz = (Dinodata.sum(axis=1)!=0)
    FlonsDino[FlonsDino<0] += 360
    FlonsDino = FlonsDino[idxgz]
    FlatsDino = FlatsDino[idxgz]
    Dinodata = Dinodata[idxgz]
    
    idx = np.where(FlatsDino<maxlatS)[0]
    FlonsDino = FlonsDino[idx]
    FlatsDino = FlatsDino[idx]
    Dinodata = Dinodata[idx]
    
#    print(np.where(np.logical_and(FlonsDino>300, FlatsDino<-50)))
    
    i1 = 241
    i2 = 230
    station1 = stations[i1]; station2 = stations[i2];
    loc1 = [FlonsDino[i1], FlatsDino[i1]]
    loc2 = [FlonsDino[i2], FlatsDino[i2]]
    Dinodata1 = Dinodata[i1]
    Dinodata2 = Dinodata[i2]
    subs = np.logical_or(Dinodata1>0, Dinodata2>0)
    Dinodata1 = Dinodata1[subs];
    Dinodata2 = Dinodata2[subs];
    species = species[subs]
    print(species)
    
    c = 18
    minlon = min(loc1[0], loc2[0]) - c - 15
    maxlon = max(loc1[0], loc2[0]) + c - 5
    minlat = max(min(loc1[1], loc2[1]) - c, -75)
    maxlat = max(loc1[1], loc2[1]) + c - 5
    exte=[minlon, maxlon, minlat, maxlat]
    #%%
    #Optics results
    f = plt.figure(constrained_layout=True, figsize = (11, 15))
    gs = f.add_gridspec(3, 10)
    gs.update(wspace=0.2)#, hspace=0.05) # set the spacing between axes. 
    #%%
    from matplotlib import cm
    import matplotlib.colors as colors
    
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap
    
    firstcloudcolor = 'k'
    xL = -30; yL = -30;
    sigma = 9
    sigma2 = 8
    bias = 10
    res = 3
    con = 3
    n = 8
    
    np.random.seed(1)
    x1 = np.random.normal(xL+bias,sigma2,n) + 12*con
    y1 = np.random.normal(yL,sigma2+2,n) + 16

    ax = f.add_subplot(gs[0, :], projection='3d')
    ax.invert_zaxis()
    
    x = np.linspace(-40, 25, 10)
    y = np.linspace(-40, 25, 10)
    X, Y = np.meshgrid(x, y)
    Z = 1.5+np.random.rand(X.shape[0],X.shape[1])/3.
    ax.plot_surface(X, Y, Z, cmap=truncate_colormap(cm.Reds, 0.3, 1),
                           linewidth=0, antialiased=False, vmin=1.5, vmax=1.75, alpha=0.3, zorder=-100)
    plt.xlim(-40,25)
    plt.ylim(-40,25)
    ax.set_xlabel('$^{\circ}$E', fontsize=18)
    ax.set_ylabel('$^{\circ}$N', fontsize=18)
    ax.set_zlabel('depth (km)', fontsize=18)
    
    
    leng = 20
    xs = np.linspace(xL+20, x1[1], leng) + 10*np.sin(np.linspace(0,4*np.pi,leng))
    ys = np.linspace(yL, y1[1], leng) + 1*np.cos(np.linspace(0,4*np.pi,leng))
    zs = np.linspace(0.9, 0, leng)+ 0.1*np.sin(np.linspace(0,2*np.pi,leng))
    
    ax.plot(xs, ys, zs,':', color='k', linewidth = 2, zorder = 10)
    
    ax.scatter3D(x1,y1, color=firstcloudcolor,alpha=1, s=50, label='first distribution')
    ax.scatter3D(xL+17,yL, [0.9], color='navy', marker = 'P', s=255, label = 'release location', zorder=10)
    
    ax.zaxis.set_ticks([1,0])
    ax.zaxis.set_ticklabels([1,0])
    
    ax.set_yticks([])
    ax.set_xticks([])
                       
    legend_el = [
                 Line2D([0], [0], linestyle='-', color='w', marker='P', markersize=18,
                        markerfacecolor='navy',
                        label='Release location'), 
                Line2D([0], [0],  color='w', marker='o',markerfacecolor='k', markersize=9 ,
                        label='particle distribution'), 
                 Line2D([0], [0], linestyle=':', color='k', linewidth=3,
                         label='trajectory'), 
                 ]
    first_legend = ax.legend(handles=legend_el, fontsize=fs-2, 
                              loc='upper left', bbox_to_anchor=(1., 0.7),ncol=1)
    
    ax.set_title('(a)', fontsize=fs)
#%%
    gs_s = [[[0,1],[0,2]], [[1,1], [1,2]], [[2,1], [2,2]], [[3,1],[3,2]], [[4,1],[4,2]]]
    panel_labels = [['(a) ','(b) ','(c) '],['(d) ', '(e) ','(f) '],['(g) ', '(h) ','(i) '],['(j) ', '(k) ','(l) ']]
    
    labels = np.array(labels)[0]
    labels[lat0>maxlat] = -1
    
    Dinoargs = nwf.find_nearest_args(lon0, lat0, FlatsDino, FlonsDino)
    reachDinodata = reachability[Dinoargs]
    Dinolabels=labels[Dinoargs]
       
    idx = (Dinolabels!=-1)
    idx0 = (Dinolabels==-1)
    labelsD = Dinolabels[idx]
    
    tot_clus = np.unique(labels)
    
    #Reachability plot

    ax = f.add_subplot(gs[1,:],projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, zorder=0, color=noisecolor)
    g = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
    g.xlocator = mticker.FixedLocator([-90, -70, -20, -0, 90, 180])
    g.xlabels_bottom = False
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
            p = ax.scatter(lon0[w0], lat0[w0], s=60, c=colorsg[li],
                              alpha=alpha, zorder=9, marker='s') 
    ax.scatter(np.array([loc1[0]]), np.array([loc1[1]]), s=300, marker='X',
               color='k', zorder=200)
    ax.scatter(np.array([loc2[0]]), np.array([ loc2[1]]), s=300, marker='D',
               color='k', zorder=200)
    ax.set_title(panel_labels[0][1], size=10, fontsize=fs)
    ax.add_feature(cfeature.LAND, zorder=100, color='peachpuff')
    ax.add_feature(cfeature.COASTLINE, zorder=100)

    # Add a scatter of the sediment sample sites
    custom_lines = [Line2D([0], [0], marker='X', markerfacecolor='k', 
                           markeredgecolor='k', 
                           lw=0, markersize=15),
                    Line2D([0], [0], marker='D', markerfacecolor='k', 
                           markeredgecolor='k', 
                           lw=0, markersize=15),
                    Line2D([0], [0], marker='s', markerfacecolor=colorsg[2], 
                           markeredgecolor=colorsg[2], 
                           lw=0, markersize=12),
                    Line2D([0], [0], marker='s',markerfacecolor=colorsg[4], 
                           markeredgecolor=colorsg[4], 
                           lw=0, markersize=12),
                    Line2D([0], [0], marker='s', markerfacecolor=colorsg[1], 
                           markeredgecolor=colorsg[1], 
                           lw=0, markersize=12)]
    
    legend = ax.legend(custom_lines, ['clustered site', 'noisy site', 'cluster 1',
                                      'cluster 2', 'cluster 3'], 
                       bbox_to_anchor=(1.7, 1.), loc='upper right', ncol=1,
                       facecolor='dimgrey', edgecolor='k', fontsize=fs-2)
    for text in legend.get_texts():
        text.set_color("snow")


#%%The pie charts \x1B[3mHello World\x1B[23m
    from matplotlib import rc
    rc('text', usetex=True)
    def make_italic(ar):
        res = []
        for a in ar:
            res.append(a+'}')
        return res
    species = np.array([r"\textit{I. aculeatum}",r"\textit{I. sphaericum}",
               r"\textit{N. labyrinthus}",r"\textit{O. centrocarpum}",
               r"\textit{S. ramosus}",r"\textit{P. dalei}",
               r"\textit{Brigantedinium spp.}",r"\textit{D. chathamensis}", 
               r"\textit{S. antarctica}"])
    piecolors = np.array(sns.color_palette("tab10", len(species)))[::-1]
    print('station clustered: ',station2)
    print('station noisy: ',station1)
    
    print('taxonomical distance versus clusters')
    ax = f.add_subplot(gs[2,5:])
    wedges, texts =  ax.pie(x=Dinodata2, colors=piecolors,
                           pctdistance=0.5,#explode = [00.01]*len(Dinodata2),
                           wedgeprops = {'linewidth': 2, 
                                         'edgecolor':'k'})
    ax.axis('equal')
    ax.set_title('(d) noisy site', fontsize=fs)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fs)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)

    legend = ax.legend(wedges, species,
          title="Species composition",
          loc="center left",
          bbox_to_anchor=(1.05, 0.03, 0.5, 1), prop={'size':fs-6})
    plt.setp(legend.get_title(),fontsize=fs)
#    ax.text(1.3,0.03,'taxonomy', fontdict={'size':fs, 'color':'k'})

    ax = f.add_subplot(gs[2,:5])   
    ax.pie(x=Dinodata1[Dinodata1>0], colors=piecolors[Dinodata1>0],
           pctdistance=0.5,#explode = [00.02]*len(Dinodata1[Dinodata1>0])
                           wedgeprops = {'linewidth': 2, 
                                         'edgecolor':'k'})
    ax.axis('equal')
    
    ax.set_title('(c) clustered site', fontsize=fs)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fs)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)
        
        
        
    #%% Save the figure
    f.savefig("figs/illustrateOPTICS.png", bbox_inches='tight')
    plt.show()
        
        
print('shannon diversity of respectively the \
clustered and noisy site: \n%.4f\n%.4f'%(shannon(Dinodata1),shannon(Dinodata2)))

