"""
This scrpt applies the CCA analysis to random equally sized subsamples of the
sedimentary data. 
"""

import os
assert os.environ['CONDA_DEFAULT_ENV']=='skbio_env', 'You should use the skbio_env conda environment here'
import numpy as np
from skbio.stats.ordination import cca
import pandas as pd
from copy import copy
from random import sample, seed

sp= 6 # sinking speed
mins = 300 # s_min
its = 10000 # Amount of iterations
seed(28) # Apply a seed for reproducibility
# Set the xi parameter
if(sp==6):
    if(mins==300):
        opts = ["xi", 0.002]
elif(sp==25):
    if(mins==300):
        opts = ["xi", 0.003]
    elif(mins==400):
        opts = ["xi", 0.003]

ff = np.load('loops/prep_CCA_sp%d_smin%d%s_%.5f.npz'%(sp, mins, opts[0], opts[1]))
#%%
# keep track of the dinocyst results
Dinolabels = ff['Dinolabels']
Dinoenv = ff['Dinoenv']
Dinoenv_nn = ff['Dinoenv_nn']
Dinomaxs = {}; Fmaxs = {};
envs = ff['envnames']
envsplt = ['temp','N','salt','Si','P']
Flabels = ff['Flabels']
Fenv = ff['Fenv']
Fenv_nn = ff['Fenv_nn']
CCA2res = []
CCA1res = []
varss = {}
for en in envsplt: 
    varss[en] = [];

#%% Apply random subsampling
# keep track of the results
varD_explained = []
varF_explained = []
for j in range(its): # loop over iterations numbers
    #Apply CCA to the dinocyst data
    data = copy(ff['Dinodata'])
    sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
    species =  np.array(['species %d'%(i) for i in range(data.shape[1])])
    
    leftd = np.sum((Dinolabels!=-1)) # the number of sites in clusters
    idleft = sample(range(len(Dinolabels)), leftd) # random subsample with this number (leftd)

    # Set up dataframes for CCA analysis    
    X = pd.DataFrame(data[idleft], sites[idleft], species)
    Y = pd.DataFrame(Dinoenv[idleft], sites[idleft], envs)
    Y_nn = pd.DataFrame(Dinoenv_nn[idleft], sites[idleft], envs)    

    # variables to delete from analysis
 #       del Y['N']
    del Y['P']
    del Y['Si']
    #del Y['temp']
    del Y['salt'] 
    
    CCA = cca(Y,X) # CCA analysis
    # keep track of explained variance at the iterations
    varD_explained.append(CCA.proportion_explained[0])
    
    #%%    
    data = ff['data']
    sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
    species =  np.array(['species %d'%(i) for i in range(data.shape[1])])

    left = np.sum((Flabels!=-1)) # the number of sites in clusters
    idleft = sample(range(len(Flabels)), left) # random subsample with this number (leftd) 
    
    # Set up dataframes for CCA analysis 
    X = pd.DataFrame(data[idleft], sites[idleft], species)
    Y = pd.DataFrame(Fenv[idleft], sites[idleft], envs)
    Y_nn = pd.DataFrame(Fenv_nn[idleft], sites[idleft], envs)
    
    # variables to delete from analysis
 #   del Y['N']
    del Y['Si']
    del Y['P']
    #del Y['temp']
    del Y['salt']
    
    CCA = cca(Y,X)# CCA analysis
    # keep track of explained variance at the iterations
    varF_explained.append(CCA.proportion_explained[0])
    
#%% Also apply the CCA if the clustering is applied
for ni,n in enumerate([False]):

    data = copy(ff['Dinodata'])
    sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
    species =  np.array(['species %d'%(i) for i in range(data.shape[1])])
    
    args = np.where(Dinolabels!=-1)
    data = data[args]
    sites = sites[args]
    Dinoenv = Dinoenv[args]
    Dinoenv_nn = Dinoenv_nn[args]
    
    X = pd.DataFrame(data, sites, species)
    Y = pd.DataFrame(Dinoenv, sites, envs)
    Y_nn = pd.DataFrame(Dinoenv_nn, sites, envs)    
    
    #del Y['N']
    del Y['Si']
    del Y['P']
    #del Y['temp']
    del Y['salt']  
    
    CCA = cca(Y,X)  
    dinovar = CCA.proportion_explained[0]
   
    data = ff['data']
    sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
    species =  np.array(['species %d'%(i) for i in range(data.shape[1])])

    args = np.where(Flabels!=-1)
    data = data[args]
    sites = sites[args]
    Fenv = Fenv[args]
    Fenv_nn = Fenv_nn[args]
    
    X = pd.DataFrame(data, sites, species)
    Y = pd.DataFrame(Fenv, sites, envs)
    Y_nn = pd.DataFrame(Fenv_nn, sites, envs)
    
    #del Y['N']
    del Y['Si']
    del Y['P']
    #del Y['temp']
    del Y['salt']
    
    CCA = cca(Y,X)
    fvar = CCA.proportion_explained[0]
#%%
print('fraction of %d iterations which explained more variance than the clustered sites'%(its))
print('Dinos: ',np.sum(np.array(varD_explained)>dinovar) / its)
print('Forams: ',np.sum(np.array(varF_explained)>fvar) / its)
print('fraction of total (dinos and forams): ',leftd/len(Dinolabels),'   ', left/len(Flabels))
