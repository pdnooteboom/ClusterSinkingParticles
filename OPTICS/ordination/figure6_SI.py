import os
assert os.environ['CONDA_DEFAULT_ENV']=='skbio_env', 'You should use the conda environment skbio_env'
import numpy as np
from skbio.stats.ordination import cca
import pandas as pd
import matplotlib.pylab as plt
from copy import copy
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Patch

redFspecies = True

spl = [6,11,25,250]
tits = ['(a)', '(b)', '(c)', '(d)']
Allvars = False
noise = [True,False]
plott=True#False#
dirRead = '/Users/nooteboom/Documents/GitHub/cluster_TM/cluster_SP/density/dens/ordination/'

minss = [100,200, 300, 400, 500, 600, 700, 800, 900,1000] # The s_min values
xiss = np.arange(0.0001,0.01, 0.0001) # The xi values

fig, ax = plt.subplots(2,3, figsize=(16,16),
                       gridspec_kw={'width_ratios':[1,1,0.08]})
ax[0,0].get_shared_y_axes().join(ax[1,0])
ax[0,1].get_shared_y_axes().join(ax[1,1])
for axs in ax[:, 2]:
    axs.remove()
gs = ax[1, 2].get_gridspec()
axbig = fig.add_subplot(gs[:, 2])
sns.set(style='whitegrid',context='paper', font_scale=2)
fs=20
vs = np.array([-1,1])*0.8

for spi, sp in enumerate(spl):
    print(sp)
    # keep track of the results
    # F and D stand for Foram and Dino
    # noise keeps track of CCA results if noisy locations are included
    # cluster keeps track of results if noisy locations are excluded
    FNoise = np.zeros((len(minss), len(xiss)))
    DNoise = np.zeros((len(minss), len(xiss)))
    FCluster = np.zeros((len(minss), len(xiss)))
    DCluster = np.zeros((len(minss), len(xiss)))
    
    for mini,mins in enumerate(minss):
        print('min:  %d'%(mins))
        for xii, xis in enumerate(xiss):
            opts = ["xi", xis]
    
            if(redFspecies):
                ff = np.load('loops/redF/prepredF_CCA_sp%d_smin%d%s_%.5f.npz'%(sp, mins, opts[0], opts[1]))       
            else:
                ff = np.load(dirRead+'loops/prep_CCA_sp%d_smin%d%s_%.5f.npz'%(sp, mins, opts[0], opts[1]))
            #%%
            envs = ff['envnames']
            if(Allvars):   
                envsplt = ff['envnames']    
            else:  
                envsplt = ff['envnames']    
                envsplt = ['temp','N']
            Flabels = ff['Flabels']
            Flabelsfull = copy(Flabels)
            Fenv = ff['Fenv']
            for ni,n in enumerate(noise):
                envs = ff['envnames']
                envsplt = ['temp','N']
                Flabels = ff['Flabels']
                Fenv = ff['Fenv']
                Fenv_nn = ff['Fenv_nn']
    
                #%% Foraminifera
                data = ff['data']
                sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
                species =  np.array(['species %d'%(i) for i in range(data.shape[1])])
                
                if(not n):
                    args = np.where(Flabels!=-1)
                    data = data[args]
                    Flabels = Flabels[args]
                    sites = sites[args]
                    Fenv = Fenv[args]
                    Fenv_nn = Fenv_nn[args]
                
                X = pd.DataFrame(data, sites, species)
                Y = pd.DataFrame(Fenv, sites, envs)
                Y_nn = pd.DataFrame(Fenv_nn, sites, envs)
                
        #        del Y['N']
                del Y['Si']
       #         del Y['P']
                #del Y['temp']
       #         del Y['salt']
                
                if(len(Y.values)!=0):
                    if(Y.shape[0]>1):
                        CCA = cca(Y,X)
                    else:
                        FCluster[mini,xii] = np.nan
                        
                    if(n):
                        FNoise[mini,xii] = np.sum(CCA.proportion_explained[:len(CCA.proportion_explained)//2])
                    else:
                        FCluster[mini,xii] = np.sum(CCA.proportion_explained[:len(CCA.proportion_explained)//2])
                else:
                    FCluster[mini,xii] = np.nan
    #%% Load the significant according to the subsamples
    its = 999
    siglevel = 0.05
    if(redFspecies):
        ffsig = np.load('randomsubsamples_redF_sp%d_its%d.npz'%(sp,its))
    else:
        ffsig = np.load('randomsubsamples_sp%d_its%d.npz'%(sp,its))
    percF = ffsig['Fperc']
    assert percF.shape==FNoise.shape
    
    color1 = plt.cm.copper(np.linspace(0, 1, int(100*(1-siglevel))))
    color2 = plt.cm.Blues(np.linspace(0.8, 1, int(100*siglevel)))
    # combine them and build a new colormap
    cmapp = mcolors.LinearSegmentedColormap.from_list('my_colormap', np.vstack((color2, color1)))
    fs=20
    
    # Set the x ticks:
    num_ticks = len(xiss)
    frac = 6
    xticks = np.linspace(0, (len(xiss) - 1), num_ticks, dtype=np.int)
    xticklabels = []
    for i, idx in enumerate(xticks):
        if(i%frac==0):
            xticklabels.append(np.round(xiss[idx] / 1e-3,5))
        else:
            xticklabels.append('')        
    
    FN = (FCluster-FNoise)
    FN[FN==0] = np.nan
    FN = pd.DataFrame(data=FN, index=minss, columns=xiss)
    
    if(False): # if a two-sided test is used
        sigF = np.full(percF.shape, '')
        for i in range(percF.shape[0]):
            for j in range(percF.shape[1]):
                if(np.array(FN)[i,j]>=0):
                    if(percF[i,j]>siglevel/2):
                        sigF[i,j] = 'l'
                elif(np.array(FN)[i,j]<0):
                    if((1-percF[i,j])>siglevel/2):
                        sigF[i,j] = 'l'
    else: # otherwise a one-sided test is used
        sigF = (percF<=siglevel).astype(str)
        sigF[sigF=='False'] = 'l'
        sigF[sigF=='True'] = ''
    
    #%
    
    # Create the colormap
    colors2 = plt.cm.RdGy(np.linspace(0, 1, 128))[::-1]
    # combine them and build a new colormap
    cmap1 = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors2)
    cmap1.set_bad("tab:blue") # set the color for nan values
    cmap1.set_under("tab:blue") # set the color for insignificant values
    
    # Set the x ticks:
    num_ticks = len(xiss)
    frac = 6
    xticks = np.linspace(0, (len(xiss) - 1), num_ticks, dtype=np.int)
    xticklabels = []
    for i, idx in enumerate(xticks):
        if(i%frac==0):
            xticklabels.append(np.round(xiss[idx] / 1e-3,5))
        else:
            xticklabels.append('')
            
    
    DN = (DCluster-DNoise)
    FN = (FCluster-FNoise)
    FN[FN==0] = np.nan
    DN[DN==0] = np.nan
    
    DN = pd.DataFrame(data=DN, index=minss, columns=xiss)
    FN = pd.DataFrame(data=FN, index=minss, columns=xiss)
    
    # The figure
    
    if(spi==0):
        g1 = sns.heatmap(FN,cmap=cmap1,ax=ax[spi//2, spi%2], vmin=vs[0], vmax=vs[1], 
                     xticklabels=xticklabels, annot=sigF, fmt='', cbar_ax=axbig)
    else:
        g1 = sns.heatmap(FN,cmap=cmap1,ax=ax[spi//2, spi%2], vmin=vs[0], vmax=vs[1], 
                     xticklabels=xticklabels, annot=sigF, fmt='', cbar=False)
    ax[spi//2, spi%2].set_yticklabels(minss,fontsize=fs-6, rotation='horizontal')
    if(spi%2==0):
        g1.set_ylabel('$s_{min}$', fontsize=fs)
    else:
        g1.set_yticklabels([])
    if(spi//2==1):
        ax[spi//2, spi%2].set_xticklabels(xticklabels, fontsize=fs-6, rotation='vertical')
        ax[spi//2, spi%2].set_xticks(xticks+0.5)
        g1.set_xlabel('$\\xi\cdot10^{-3}$', fontsize=fs)
    else:
        g1.set_xticklabels([])
        g1.set_xticks([])
    g1.set_title(tits[spi], fontsize=fs)

legend_elements = [Patch(facecolor=cmap1(np.nan),
                         label='-no cluster\n', linewidth=0)]
leg = axbig.legend(handles=legend_elements, loc='upper left',
             bbox_to_anchor=(-0.525, -0.1005, 0.1, 0.1), frameon=False,
             handletextpad=-0.25)

for patch in leg.get_patches():
    patch.set_height(41)
    patch.set_width(30)
    
 #   g2 = sns.heatmap(FN,cmap=cmap1,cbar=True,ax=ax[1],  vmin=-0.35, vmax=0.35, 
 #                    cbar_ax=ax[0,2], yticklabels=False, annot=sigF, fmt='')
 #   ax[1].set_xticks(xticks+0.5)
 #   ax[1].set_xticklabels(xticklabels, fontsize=fs-6, rotation='vertical')
    
plt.savefig('heatmap_SI_CCA.png', dpi=300,bbox_inches='tight')
plt.subplots_adjust(hspace=0.07)
plt.show()