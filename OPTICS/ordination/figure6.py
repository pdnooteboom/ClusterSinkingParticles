import os
assert os.environ['CONDA_DEFAULT_ENV']=='skbio_env', 'You should use the conda environment skbio_env'
import numpy as np
from skbio.stats.ordination import cca
import pandas as pd
import matplotlib.pylab as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from copy import copy
import matplotlib.colors as mcolors
def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

def logtransform(data):
    data *= 10**4 
    return np.log(data+1)

sp = 6
Allvars = False
noise = [True,False]
plott=True#False#
logtransform_data = False
dirRead = '/Users/nooteboom/Documents/GitHub/cluster_TM/cluster_SP/density/dens/ordination/'

minss = [100,200, 300, 400, 500, 600, 700, 800, 900,1000] # The s_min values
xiss = np.arange(0.0001,0.01, 0.0001) # The xi values

VIFt = 3.06 # Put a threshold on the VIF

# keep track of the results
# F and D stand for Foram and Dino
# noise keeps track of CCA results if noisy locations are included
# cluster keeps track of results if noisy locations are excluded
# VIF keep track of the variance inflation factor (VIF)
FNoise = np.zeros((len(minss), len(xiss)))
DNoise = np.zeros((len(minss), len(xiss)))
FCluster = np.zeros((len(minss), len(xiss)))
DCluster = np.zeros((len(minss), len(xiss)))
DVIF = np.zeros((len(minss), len(xiss)))
FVIF = np.zeros((len(minss), len(xiss)))

for mini,mins in enumerate(minss):
    print('min:  %d'%(mins))
    for xii, xis in enumerate(xiss):
        opts = ["xi", xis]

        ff = np.load(dirRead+'loops/prep_CCA_sp%d_smin%d%s_%.5f.npz'%(sp, mins, opts[0], opts[1]))
        #%%
        
        Dinolabels = ff['Dinolabels']
        Dinolabels_full = copy(Dinolabels)
        Dinoenv = ff['Dinoenv']
        Dinomaxs = {}; Fmaxs = {};
        envs = ff['envnames']
        if(Allvars):   
            envsplt = ff['envnames']    
        else:  
            envsplt = ff['envnames']    
            envsplt = ['temp','N']
        Flabels = ff['Flabels']
        Flabelsfull = copy(Flabels)
        Fenv = ff['Fenv']
        CCA2res = []
        CCA1res = []
        varss = {}
        for en in envsplt: 
            varss[en] = [];
            
        name = []
        
        Dinoname = []
        DinoCCA2res = []
        DinoCCA1res = []
        Dinovarss = {}
        for en in envsplt: Dinovarss[en] = [];
        Dinoname = []
        
        var_explained = []
        tabel_t = []#np.zeros((len(noise), 2))
        tabel_tD = []#np.zeros((len(noise), 2))
        tabel_tF = []#np.zeros((len(noise), 2))
        for ni,n in enumerate(noise):
            Dinolabels = ff['Dinolabels']
            Dinoenv = ff['Dinoenv']
            Dinoenv_nn = ff['Dinoenv_nn']
            envs = ff['envnames']
            envsplt = ['temp','N']
            Flabels = ff['Flabels']
            Fenv = ff['Fenv']
            Fenv_nn = ff['Fenv_nn']
#%% Dinos
            data = ff['Dinodata']
            sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
            species =  np.array(['species %d'%(i) for i in range(data.shape[1])])
            
            if(not n):
                args = np.where(Dinolabels!=-1)
                data = data[args]
                Dinolabels = Dinolabels[args]
                sites = sites[args]
                Dinoenv = Dinoenv[args]
                Dinoenv_nn = Dinoenv_nn[args]
            
            if(logtransform_data):
                data = logtransform(data)
            
            X = pd.DataFrame(data, sites, species)
            Y = pd.DataFrame(Dinoenv, sites, envs)
            Y_nn = pd.DataFrame(Dinoenv_nn, sites, envs)    
            # in order of highest eienvalue:
     #       del Y['N']
            del Y['Si']
      #      del Y['P']
            #del Y['temp']
      #      del Y['salt']

            if(len(Y.values)!=0):
            
                if(len(Y.columns)>1):
                    if((calc_vif(Y)['VIF']>VIFt).any()):
                        DVIF[mini,xii] = 1                
                
                CCA = cca(Y,X)
    
                if(n):
                    DNoise[mini,xii] = np.sum(CCA.proportion_explained[:len(CCA.proportion_explained)//2])
                else:
                    DCluster[mini,xii] = np.sum(CCA.proportion_explained[:len(CCA.proportion_explained)//2])
            else:
                DCluster[mini,xii] = np.nan

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
            
            if(logtransform_data):
                data = logtransform(data)
            
            X = pd.DataFrame(data, sites, species)
            Y = pd.DataFrame(Fenv, sites, envs)
            Y_nn = pd.DataFrame(Fenv_nn, sites, envs)
            
    #        del Y['N']
            del Y['Si']
   #         del Y['P']
            #del Y['temp']
   #         del Y['salt']
            
            if(len(Y.values)!=0):        
                if(len(Y.columns)>1):
                    if((calc_vif(Y)['VIF']>VIFt).any()):
                        FVIF[mini,xii] = 1
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

#%%
import seaborn as sns
sns.set(style='whitegrid',context='paper', font_scale=2)
fs=20


# Create the colormap
colors2 = plt.cm.RdGy(np.linspace(0, 1, 128))[::-1]
# combine them and build a new colormap
cmap1 = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors2)
cmap1.set_bad("tab:blue")

# Set the x ticks:
num_ticks = len(xiss)
frac = 6
xticks = np.linspace(0, (len(xiss) - 1), num_ticks, dtype=np.int)
xticklabels = []
for i, idx in enumerate(xticks):
    if(i%frac==0):
        xticklabels.append(np.round(xiss[idx],5))
    else:
        xticklabels.append('')
        

DN = (DCluster-DNoise)
FN = (FCluster-FNoise)
FN[FN==0] = np.nan
DN[DN==0] = np.nan

DN = pd.DataFrame(data=DN, index=minss, columns=xiss)
FN = pd.DataFrame(data=FN, index=minss, columns=xiss)


# The figure
fig, ax = plt.subplots(1,3, figsize=(16,8),
                       gridspec_kw={'width_ratios':[1,1,0.08]})
ax[0].get_shared_y_axes().join(ax[1])
g1 = sns.heatmap(DN,cmap=cmap1,cbar=False,ax=ax[0], vmin=-0.35, vmax=0.35, 
                 xticklabels=xticklabels)
ax[0].set_xticklabels(xticklabels, fontsize=fs-6, rotation='vertical')
ax[0].set_xticks(xticks+0.5)
ax[0].set_yticklabels(minss,fontsize=fs-6, rotation='horizontal')
g1.set_ylabel('$s_{min}$', fontsize=fs)
g1.set_title('(a) dinocysts', fontsize=fs)
g1.set_xlabel('$\\xi$', fontsize=fs)


g2 = sns.heatmap(FN,cmap=cmap1,cbar=True,ax=ax[1],  vmin=-0.35, vmax=0.35, 
                 cbar_ax=ax[2], yticklabels=False)
ax[1].set_xticks(xticks+0.5)
ax[1].set_xticklabels(xticklabels, fontsize=fs-6, rotation='vertical')
g2.set_title('(b) foraminifera', fontsize=fs)
g2.set_xlabel('$\\xi$', fontsize=fs)

plt.savefig('heatmap_CCA_sp%d.png'%(sp), dpi=300,bbox_inches='tight')
plt.show()