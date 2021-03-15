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
from random import sample, seed, randint
from time import time

dirRead = '/Users/nooteboom/Documents/GitHub/cluster_TM/cluster_SP\
/density/dens/ordination/'

dirRead = '/Users/nooteboom/Documents/GitHub/cluster_TM/cluster_SP/density/dens/ordination/'
sp= 25 # sinking speed
its = 999 # Amount of iterations
print('its: ',its)
minss = [100,200,300,400,500,600,700,800,900,1000]
xiss = np.arange(0.0001,0.01, 0.0001)

mincl = 3 # minimum of allowed clustered samples

Dperc = np.full((len(minss), len(xiss)), -1.)
Fperc = np.full((len(minss), len(xiss)), -1.)

ti = time()
for mini, mins in enumerate(minss): # loop the s_min values
    print(mini/len(minss))
    for xisi, xis in enumerate(xiss): # loop the xi parameters
        opts = ["xi", xis]
        ff = np.load(dirRead + 'loops/prep_CCA_sp%d_smin%d%s_%.5f.npz'%(sp, mins, opts[0], opts[1])) # load cca results
        #%%
        # keep track of the dinocyst results
        Dinolabels = ff['Dinolabels']
        envs = ff['envnames']
        envsplt = ['temp','N','salt','Si','P']
        Flabels = ff['Flabels']
        
        #%% Apply random subsampling
        # keep track of the results
        varD_explained = []
        #%% Apply the CCA if the clustering is applied
        args = np.where(Dinolabels!=-1)
        if(len(args[0])>mincl):
            Dinoenv = copy(ff['Dinoenv'])
            Dinoenv_nn = copy(ff['Dinoenv_nn'])
            data = copy(ff['Dinodata'])
            sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
            species =  np.array(['species %d'%(i) for i in range(data.shape[1])])
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

            seed(28) # Apply a seed for reproducibility of the random iterations
            for j in range(its): # loop over iterations numbers  
 #               seed(j)              
                #Apply CCA to the a random subsample of dinocyst data
                Dinoenv = copy(ff['Dinoenv'])
                Dinoenv_nn = copy(ff['Dinoenv_nn'])
                data = copy(ff['Dinodata'])
                sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
                species =  np.array(['species %d'%(i) for i in range(data.shape[1])])
                
                leftd = randint(mincl,len(sites)-mincl) # the number of sites in random clusters
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
            #%%    The same for the foraminifera

        varF_explained = []
        Fenv = copy(ff['Fenv'])
        Fenv_nn = copy(ff['Fenv_nn'])
        data = copy(ff['data'])
        sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
        species =  np.array(['species %d'%(i) for i in range(data.shape[1])])
        
        args = np.where(Flabels!=-1)
        if(len(args[0])>mincl):
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
            
            seed(28) # Apply a seed for reproducibility of the random iterations
            for j in range(its): # loop over iterations numbers
 #               seed(j)
            
                Fenv = copy(ff['Fenv'])
                Fenv_nn = copy(ff['Fenv_nn'])
                data = copy(ff['data'])
                sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
                species =  np.array(['species %d'%(i) for i in range(data.shape[1])])
            
                left = randint(mincl,len(sites)-mincl) # the number of sites in random clusters
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
        if(np.isnan(dinovar)):
            Dperc[mini,xisi] = -1
        else:
            Dperc[mini, xisi] = np.sum(np.array(varD_explained)>dinovar) / its
    
        if(np.isnan(fvar)):
            Fperc[mini,xisi] = -1
        else:
            Fperc[mini, xisi] = np.sum(np.array(varF_explained)>fvar) / its
#            print('F ',np.sum(np.array(varF_explained)>dinovar) / its)
print('time (seconds): ',time()-ti)
#%%
np.savez('permutationtest_sp%d_its%d.npz'%(sp,its), 
         mins=mins, xis=xis, Dperc=Dperc, Fperc=Fperc)

#%%
import matplotlib.pylab as plt

plt.figure(figsize=(10,10))
plt.imshow(Dperc, vmin=0, vmax=0.05)
plt.colorbar()
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(Fperc, vmin=0, vmax=0.05)
plt.colorbar()
plt.show()
