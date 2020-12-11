import os
assert os.environ['CONDA_DEFAULT_ENV']=='skbio_env', 'You should use the skbio_env conda environment here'
import numpy as np
from skbio.stats.ordination import cca
import pandas as pd
import matplotlib.pylab as plt
from copy import copy
import statsmodels.api as sm
import seaborn as sns
import plot_functions as plf
sns.set(context='paper', style='whitegrid')

#%% parameters    
sp= 6 # sinking speed
mins = 300 # s_min parameter
# The xi:
if(sp==6):
    if(mins==400):
        opts = ["xi", 0.0025]#05]
    #    opts = ["xi", 0.0026]#05]
    elif(mins==200):
        opts = ["xi", 0.0045]
    elif(mins==300):
        opts = ["xi", 0.002]
if(sp==25):
    if(mins==300):
        opts = ["xi", 0.01]
    if(mins==400):
        opts = ["xi", 0.03]

ff = np.load('loops/prep_CCA_sp%d_smin%d%s_%.5f.npz'%(sp, mins, opts[0], opts[1]))
Allvars = False
noise = [True,False]
bvif = False

# For plotting: 
fs = 20 # fontsize
s=35 # markersize
s2 = 25 # size for plotting
templim = [-3,30] # temperature y-axis limits
Nlim = [-0.3, 28] # nitrate y-axis limits
#%%
# create the colors for plotting
colo = ["gist_ncar","Greys"]#"Purples","Blues","Greens","Reds"]#"Oranges",
colorsg = []
its = 13#70
colorsg.append(sns.color_palette(colo[1], n_colors=its+1)[-1])
for k in range(its):
    colorsg.append(sns.color_palette(colo[0], n_colors=its+1)[k])
colorsg.reverse()
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
    envsplt = ['temp','N','salt','Si','P']
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
tabel_t = []
tabel_tD = []
tabel_tF = []
CCAobj = []
for ni,n in enumerate(noise):
    print('\n\n')
    print('The fdinocysts')
    Dinolabels = ff['Dinolabels']
    Dinoenv = ff['Dinoenv']
    Dinoenv_nn = ff['Dinoenv_nn']
    envs = ff['envnames']
    envsplt = ['temp','N']
    Flabels = ff['Flabels']
    Fenv = ff['Fenv']
    Fenv_nn = ff['Fenv_nn']

    print('Dinos')

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
    
    X = pd.DataFrame(data, sites, species)
    Y = pd.DataFrame(Dinoenv, sites, envs)
    Y_nn = pd.DataFrame(Dinoenv_nn, sites, envs)    
    # in order of highest eienvalue:

    if(Allvars):
        print('All variables used')
    elif(n): # delete variables from analysis
        print('')
 #       del Y['N']
        del Y['Si']
        del Y['P']
        #del Y['temp']
        del Y['salt']
    else:
        print('')
 #       del Y['N']
        del Y['Si']
        del Y['P']
        #del Y['temp']
        del Y['salt']
       
    CCA = cca(Y,X)
    CCAobj.append(CCA)
    
    Dinoname.append('noise: '+str(n))
    DinoCCA2res.append(CCA.samples['CCA2'])
    DinoCCA1res.append(CCA.samples['CCA1'])
    for en in Y.columns:
        Dinovarss[en].append(Y_nn[en])
    
    if(Allvars):
        ress = []
        ivn = []
        for eni,en in enumerate(envs):
            if(en in Y.columns):
                ivn.append(en)
                res = 0
                for col in CCA.features.columns[:len(CCA.features.columns)//2]:
                    res += np.abs(CCA.features[col][en] * CCA.eigvals[col])
                ress.append(res)
        ress = np.array(ress)
    else:
        ivn = []
        for eni,en in enumerate(envs):
            if(en in Y.columns):
                ivn.append(en)
                res = 0
                for col in CCA.features.columns[:len(CCA.features.columns)//2]:
                    res += np.abs(CCA.features[col][en] * CCA.eigvals[col])
                print('%s    %.5f'%(en, res))
    var_explained.append(CCA.proportion_explained)
    tabel_tD.append(['%.9f'%(CCA.proportion_explained[i]) for i in range(2)])#len(CCA.proportion_explained)//2)])
    plf.CCAplot(CCA, Y, ivn, clus=Dinolabels, dataset='Dinos')
    #%%
    print('\n\n')
    print('The foraminifera')
    
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
    
    if(Allvars):
        print('All variables used, noise ',n)
    elif(n):
        print('')
 #       del Y['N']
        del Y['Si']
        del Y['P']
        #del Y['temp']
        del Y['salt']
    else:
        print('')
#        del Y['N']
        del Y['Si']
        del Y['P']
        #del Y['temp']
        del Y['salt']
    
    CCA = cca(Y,X)
    CCAobj.append(CCA)
    
    name.append('noise: '+str(n))
    CCA2res.append(CCA.samples['CCA2'])
    CCA1res.append(CCA.samples['CCA1'])
    for en in Y.columns:
        varss[en].append(Y_nn[en])
    
    if(Allvars):
        ress = []
        ivn = []
        for eni,en in enumerate(envs):
            if(en in Y.columns):
                ivn.append(en)
                res = 0
                for col in CCA.features.columns[:len(CCA.features.columns)//2]:
                    res += np.abs(CCA.features[col][en] * CCA.eigvals[col])
                ress.append(res)
        ress = np.array(ress)
    else:    
        ivn = []
        for eni,en in enumerate(envs):
            if(en in Y.columns):
                ivn.append(en)
                res = 0
                for col in CCA.features.columns[:len(CCA.features.columns)//2]:
                    res += np.abs(CCA.features[col][en] * CCA.eigvals[col])
    tabel_tF.append(['%.9f'%(CCA.proportion_explained[i]) for i in range(2)])#len(CCA.proportion_explained)//2)])
    var_explained.append(CCA.proportion_explained)
    plf.CCAplot(CCA, Y, ivn, clus=Flabels, dataset='Forams')
#%%

colls = ['first axis', ' second axis', 'third axis', 'fourth axis', 'fifth axis']
tabel_tF =   plf.square_table(tabel_tF)      
tabel_tD =   plf.square_table(tabel_tD)      
units = {'temp': ' ($^{\circ}$C)',
      'N':'O$_3$ ($\mu$mol/L)'}
tits2 = np.array([['(a) ','(b) '],['(c) ','(d) ']])
tits = np.array([['(e) ','(f) '],['(g) ','(h) ']])      


if(True): # create the figure
    fig = plt.figure(figsize=(20, 10))
    gs0 = fig.add_gridspec(1, 2)
    gs00 = gs0[1].subgridspec(2, 2)
    gs01 = gs0[0].subgridspec(2, 2)
    gs00.figure.suptitle('Dinocysts                                       '+
                         '                                                '+
                         '         Foraminifera', fontsize=fs)
    
    f00 = fig.add_subplot(gs00[0,0])
    f01 = fig.add_subplot(gs00[0,1])
    f10 = fig.add_subplot(gs00[1,0])
    f11 = fig.add_subplot(gs00[1,1])        
    
    fis = np.array([[f00, f01],[f10,f11]])
    
    for m in range(len(CCA2res)):
        for vai, va in enumerate(list(varss.keys())[:2]):
            X = sm.add_constant(np.array(CCA1res[m]))
            y = np.array(varss[va][m])
            reg = sm.OLS(y,
                         X)  
            res = reg.fit()
            plf.plot_regres(fis[vai,m],res, CCA1res[m])
            if(m==0):
                fis[vai,m].scatter(CCA1res[m][Flabelsfull<0], 
                   varss[va][m][Flabelsfull<0], s=s,marker='+', 
                   label=name[m], c='lightgray')
                for li,dl in enumerate(np.unique(Flabelsfull[Flabelsfull>0])):
                    if((Flabelsfull==dl).sum()>0):
                        fis[vai,m].scatter(CCA1res[m][Flabelsfull==dl], 
                           varss[va][m][Flabelsfull==dl], s=s+s2,marker='+', 
                           label=name[m], c=colorsg[li])
            else:
                for li,dl in enumerate(np.unique(Flabels)):
                    if((Flabels==dl).sum()>0):
                        fis[vai,m].scatter(CCA1res[m][Flabels==dl], 
                           varss[va][m][Flabels==dl], s=s+s2,marker='+', 
                           label=name[m], c=colorsg[li])
            rs =res.rsquared
            fis[vai,m].set_title(tits[vai,m] + ' RMS: %.2f'%((res.ssr/X.shape[0])), fontsize=fs)
            if(va=='temp'):
                fis[vai,m].set_ylim(templim[0],templim[1])
            elif(va=='N'):
                fis[vai,m].set_ylim(Nlim[0],Nlim[1])
            if(m>0):
                fis[vai,m].set_yticklabels([])
            if(vai==0):
                fis[vai,m].set_xticklabels([])
                fis[vai,m].set_yticks([0,10, 20,30])
                if(m==0):
                    fis[vai,m].set_yticklabels([0,10,20,30], fontsize=fs)
            if(vai>0):
                fis[vai,m].set_yticks([0,10, 20,30])
                if(m==0):
                    fis[vai,m].set_yticklabels([0,10,20,30], fontsize=fs)

            
    fis[-1,0].set_xlabel('first CCA axis', fontsize=fs)
    fis[-1,1].set_xlabel('first CCA axis', fontsize=fs)
   
#%%
    f00 = fig.add_subplot(gs01[0,0])
    f01 = fig.add_subplot(gs01[0,1])
    f10 = fig.add_subplot(gs01[1,0])
    f11 = fig.add_subplot(gs01[1,1])        
    
    fis2 = np.array([[f00, f01],[f10,f11]])
        
    for m in range(len(DinoCCA2res)):
        for vai, va in enumerate(list(Dinovarss.keys())[:2]):
            if(va=='temp'):
                fis2[vai,0].set_ylabel('SST'+units[va], fontsize=fs) 
            else:
                fis2[vai,0].set_ylabel(va+units[va], fontsize=fs) 
            X = sm.add_constant(np.array(DinoCCA1res[m]))
            y = np.array(Dinovarss[va][m])
            reg = sm.OLS(y,
                         X)  
            res = reg.fit()
            plf.plot_regres(fis2[vai,m],res, DinoCCA1res[m])
            if(m==0):
                fis2[vai,m].scatter(DinoCCA1res[m][Dinolabels_full<0], 
                   Dinovarss[va][m][Dinolabels_full<0], s=s,marker='+', 
                   label=Dinoname[m], c='lightgray')
                for li,dl in enumerate(np.unique(Dinolabels_full[Dinolabels_full>0])):
                    if((Dinolabels_full==dl).sum()>0):
                        fis2[vai,m].scatter(DinoCCA1res[m][Dinolabels_full==dl], 
                           Dinovarss[va][m][Dinolabels_full==dl], s=s+s2,marker='+', 
                           label=Dinoname[m], c=colorsg[li])
            else:
                for li,dl in enumerate(np.unique(Dinolabels_full[Dinolabels_full>0])):
                    if((Dinolabels==dl).sum()>0):
                        fis2[vai,m].scatter(DinoCCA1res[m][Dinolabels==dl], 
                           Dinovarss[va][m][Dinolabels==dl], s=s+s2,marker='+', 
                           label=Dinoname[m], c=colorsg[li])
            rs =res.rsquared
            fis2[vai,m].set_title(tits2[vai,m] + ' RMS: %.2f'%((res.ssr/X.shape[0])), fontsize=fs)
            if(va=='temp'):
                fis2[vai,m].set_ylim(templim[0],templim[1])
            elif(va=='N'):
                fis2[vai,m].set_ylim(Nlim[0],Nlim[1])
            if(m>0):
                fis2[vai,m].set_yticklabels([])
            if(vai==0):
                fis2[vai,m].set_xticklabels([])
                fis2[vai,m].set_yticks([0,10, 20,30])
                if(m==0):
                    fis2[vai,m].set_yticklabels([0,10,20,30], fontsize=fs)
            if(vai>0):
                fis2[vai,m].set_yticks([0,10, 20,30])
                if(m==0):
                    fis2[vai,m].set_yticklabels([0,10,20,30], fontsize=fs)

    fis2[-1,0].set_xlabel('first CCA axis', fontsize=fs)
    fis2[-1,1].set_xlabel('first CCA axis', fontsize=fs)

#%% To construct the tables below the figure
    plt.subplots_adjust( bottom=0.2)
    
    tabel = fis[-1,-1].table(cellText=tabel_tF,
              rowLabels=['with noise (e), (g)', 'clusters only (f),(h)'],
              colLabels=colls[:len(tabel_tF[0])],
              loc='bottom',
              bbox = [0., -0.8, 1, 0.4]
            )

    tabel.set_fontsize(fs)    

    tabel = fis2[-1,-1].table(cellText=tabel_tD,
              rowLabels=['with noise (a), (c)', 'clusters only (b), (d)'],
              colLabels=colls[:len(tabel_tD[0])],
              loc='bottom',
              bbox = [-0., -0.8, 1, 0.4]
            )
    tabel.set_fontsize(fs)
    
    
    # the titles of the tables
    
    tabel = fis[-1,-1].table(cellText=[['explained variance']],
              loc='bottom',
              bbox = [0., -0.41, 1, 0.17]
            )

    tabel.set_fontsize(fs-1)

    tabel = fis2[-1,-1].table(cellText=[['explained variance']],
              loc='bottom',
              bbox = [0., -0.41, 1, 0.17]
            )
    tabel.set_fontsize(fs-1)
#%%
    plt.savefig('Regression_CCA_sp%d_smin%d_%s%.5f.pdf'%(sp,mins,opts[0],opts[1]), bbox_inches='tight', 
                dpi=300)
    plt.show()
        
