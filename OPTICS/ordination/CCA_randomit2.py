import os
assert os.environ['CONDA_DEFAULT_ENV']=='skbio_env', 'You should use the skbio_env conda environment here'
import numpy as np
from skbio.stats.ordination import cca
import pandas as pd
from copy import copy
from random import sample, seed

sp= 6
mins = 300
its = 10000
seed(28)
if(sp==6):
    if(mins==400):
        opts = ["xi", 0.00235]#05]
    #    opts = ["xi", 0.0026]#05]
    elif(mins==200):
        opts = ["xi", 0.0015]
    #    opts = ["xi", 0.004]
    elif(mins==500):
        opts = ["xi", 0.00225]
    elif(mins==300):
        opts = ["xi", 0.002]
elif(sp==25):
    if(mins==300):
        opts = ["xi", 0.003]
        opts = ["xi", 0.0001]
    if(mins==400):
        opts = ["xi", 0.003]
#        opts = ["xi", 0.0001]

ff = np.load('loops/prep_CCA_sp%d_smin%d%s_%.5f.npz'%(sp, mins, opts[0], opts[1]))
#%%

Dinolabels = ff['Dinolabels']
Dinolabels_full = copy(Dinolabels)
Dinoenv = ff['Dinoenv']
Dinomaxs = {}; Fmaxs = {};
envs = ff['envnames']
envsplt = ['temp','N','salt','Si','P']
Flabels = ff['Flabels']
Flabelsfull = copy(Flabels)
Fenv = ff['Fenv']
CCA2res = []
CCA1res = []
varss = {}
for en in envsplt: 
    varss[en] = [];

Dinolabels = ff['Dinolabels']
Dinoenv = ff['Dinoenv']
Dinoenv_nn = ff['Dinoenv_nn']
envs = ff['envnames']
envsplt = ['temp','N','salt','Si','P']
Flabels = ff['Flabels']
Fenv = ff['Fenv']
Fenv_nn = ff['Fenv_nn']

name = []
Dinoname = []
DinoCCA2res = []
DinoCCA1res = []
Dinovarss = {}
for en in envsplt: Dinovarss[en] = [];
Dinoname = []
varD_explained = []
varF_explained = []
for j in range(its):
    for ni,n in enumerate([False]):

        data = copy(ff['Dinodata'])
        sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
        species =  np.array(['species %d'%(i) for i in range(data.shape[1])])
        
        leftd = np.sum((Dinolabels!=-1)) # the number of sites in clusters
        idleft = sample(range(len(Dinolabels)), leftd)
        
        X = pd.DataFrame(data[idleft], sites[idleft], species)
        Y = pd.DataFrame(Dinoenv[idleft], sites[idleft], envs)
        Y_nn = pd.DataFrame(Dinoenv_nn[idleft], sites[idleft], envs)    
    
        if(n):
     #       del Y['N']
            del Y['P']
            del Y['Si']
            #del Y['temp']
            del Y['salt'] 
        
        CCA = cca(Y,X)
#        assert len(CCA.proportion_explained)%2==0     
        varD_explained.append(CCA.proportion_explained[0])
        
        #%%    
        data = ff['data']
        sites =  np.array(['site %d'%(i) for i in range(data.shape[0])])
        species =  np.array(['species %d'%(i) for i in range(data.shape[1])])
    
        left = np.sum((Flabels!=-1)) # the number of sites in clusters
        idleft = sample(range(len(Flabels)), left)    
        
        X = pd.DataFrame(data[idleft], sites[idleft], species)
        Y = pd.DataFrame(Fenv[idleft], sites[idleft], envs)
        Y_nn = pd.DataFrame(Fenv_nn[idleft], sites[idleft], envs)
        
        if(n):
            del Y['N']
       #     del Y['Si']
       #     del Y['P']
            #del Y['temp']
        #    del Y['salt']
        
        CCA = cca(Y,X)
#        assert len(CCA.proportion_explained)%2==0
        varF_explained.append(CCA.proportion_explained[0])
    
#%% Also the dinovar and fvar after clustering:
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
    
 #       del Y['N']
    #del Y['Si']
    #del Y['P']
    #del Y['temp']
    #del Y['salt']  
    
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
    
 #       del Y['N']
    #del Y['Si']
    #del Y['P']
    #del Y['temp']
    #del Y['salt']
    
    CCA = cca(Y,X)
    fvar = CCA.proportion_explained[0]
#%%
#dinovar = 0.8928
#fvar = 0.80759
#if(sp== 6 and mins == 300 and opts==["xi", 0.002]):
#    dinovar = 0.865
#    fvar = 0.871
#elif(sp== 6 and mins == 400 and opts==["xi", 0.002]):
#    dinovar = 0.865
#    fvar = 0.871
#else:
#    print('g')

print('fraction of %d iterations which explained more variance than the clustered sites'%(its))
print('Dinos: ',np.sum(np.array(varD_explained)>dinovar) / its)
print('Forams: ',np.sum(np.array(varF_explained)>fvar) / its)
print('fraction of total (dinos and forams): ',leftd/len(Dinolabels),'   ', left/len(Flabels))
