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

from time import time

ti = time()

sp= 6 # sinking speed
mins = 300 # s_min
its = 50#999#10000 # Amount of iterations
# Set the xi parameter
if(sp==6):
    if(mins==300):
        opts = ["xi", 0.002]
elif(sp==25):
    if(mins==300):
        opts = ["xi", 0.003]
    elif(mins==400):
        opts = ["xi", 0.003]


dirRead = '/Users/nooteboom/Documents/GitHub/cluster_TM/cluster_SP/density/dens/ordination/'
ff = np.load(dirRead + 'loops/prep_CCA_sp%d_smin%d%s_%.5f.npz'%(sp, mins, opts[0], opts[1]))
#%%
# keep track of the dinocyst results
Dinolabels = ff['Dinolabels']
envs = ff['envnames']
envsplt = ['temp','N','salt','Si','P']
Flabels = ff['Flabels']
Fenv = ff['Fenv']
Fenv_nn = ff['Fenv_nn']
Dinoenv = ff['Dinoenv']
Dinoenv_nn = ff['Dinoenv_nn']
#%% Apply 
# keep track of the results
varD_explained = []
data = copy(ff['Dinodata'])
sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
species =  np.array(['species %d'%(i) for i in range(data.shape[1])])

args = np.where(Dinolabels!=-1)
if(len(args[0])>2):
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

#    seed(28) # Apply a seed for reproducibility
    for j in range(its): # loop over iterations numbers of random subsamples
        seed(j) # Apply a seed for reproducibility
        #Apply CCA to the dinocyst data
        Dinoenv = ff['Dinoenv']
        Dinoenv_nn = ff['Dinoenv_nn']
        data = copy(ff['Dinodata'])
        sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
        species =  np.array(['species %d'%(i) for i in range(data.shape[1])])
        
        leftd = len(args[0]) # the number of sites in clusters
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
    
#%% Apply the CCA if the clustering is applied. Foraminifera
varF_explained = []
   
data = ff['data']
sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
species =  np.array(['species %d'%(i) for i in range(data.shape[1])])

args = np.where(Flabels!=-1)
if(len(args[0])>2):
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
    print('time (seconds): ',time()-ti)    
    
    
#    seed(28) # Apply a seed for reproducibility
    for j in range(its): # loop over iterations numbers
        seed(j) # Apply a seed for reproducibility
        Fenv = ff['Fenv']
        Fenv_nn = ff['Fenv_nn']
        data = ff['data']
        sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
        species =  np.array(['species %d'%(i) for i in range(data.shape[1])])
    
        left = len(args[0]) # the number of sites in clusters
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

#%%
print('fraction of %d iterations which explained more variance than the clustered sites'%(its))
print('Dinos: ',np.sum(np.array(varD_explained)>dinovar) / its)
print('Forams: ',np.sum(np.array(varF_explained)>fvar) / its)
print('fraction of total samples left (dinos and forams): ',
      leftd/len(Dinolabels),'   ', left/len(Flabels))
