
import os
assert os.environ['CONDA_DEFAULT_ENV']=='Cartopy-py3', 'You should use the Cartopy-py3 conda environment here'
import numpy as np
from sklearn.cluster import  cluster_optics_xi, cluster_optics_dbscan
from numba import jit
from pandas import read_csv

#%% The functions
def readDinoset(name):
    # to read the surface sediment data of dinocysts
    lons = read_csv(name, usecols=[0]).values[:,0]
    lats = read_csv(name, usecols=[1]).values[:,0]
    data = read_csv(name).values[:,2:]
    data[np.isnan(data)] = 0
    row_sum = data.sum(axis=1)
    data = data / row_sum[:,np.newaxis]
    
    return lons, lats, data

# Two functions to read the surface sediment sample datasets of Foraminifera
def readForamset(name):
    file = open(name)
    data = []
    line = file.readline()
    while(len(line)>0):
        ldat = []
        j = 0
        for l in range(len(line)):
            if(line[j:l][-1:] in ['\t','\n']):
                ldat.append(line[j:l-1])
                j = l
        line = file.readline()
        data.append(ldat)
        
    data = np.array(data)
    return data

def readForamset_lonlats(name):
    file = open(name)
    data = []
    line = file.readline()
    while(len(line)>0):
        tabcount = 0
        ldat = []
        j = 0
        for l in range(len(line)):
            if(line[j:l][-1:] in ['\t','\n']):
                tabcount+=1
                if(tabcount in [5,6]):
                    ldat.append(line[j:l-1])
                j = l
        line = file.readline()
        data.append(ldat)
        
    data = np.array(data)
    return data[:,0].astype(np.float), data[:,1].astype(np.float)

def find_nearest_2a(lons, lats, lon, lat):
    dist = []
    dist = np.sqrt((lons-lon)**2+(lats-lat)**2)
    idl = np.argmin(dist)
    return lons[idl], lats[idl]

def find_nearest_args(vLons, vLats, Flats, Flons):
    args = []
    unlons = vLons
    unlats = vLats
    for i in range(len(Flons)):
        lon, lat = find_nearest_2a(unlons, unlats, Flons[i], Flats[i])
        args.append(np.where(np.logical_and(vLons==lon, vLats==lat))[0][0])
    return np.array(args)

@jit(nopython=True)
def biodiversityf(vec, typ='shannon'):
    assert len(vec.shape)==2
    # to calculate the biodiversity in the sediment samples
    if(typ=='shannon'):
        Hp = np.zeros(vec.shape[0])
        for j in range(len(Hp)):
            Hp[j] = shannon(vec[j])
        return Hp 
    else:
        Hpmax = np.zeros(vec.shape[0])
        Hp = np.zeros(vec.shape[0])
        for j in range(len(Hp)):
            if(typ=='Pielou'):
                Hp[j], Hpmax[j] = even1(vec[j])
            elif(typ=='Heip'):
                Hp[j], Hpmax[j] = even2(vec[j])
        return Hp / Hpmax

@jit(nopython=True)
def even1(vec):
    assert len(vec.shape)==1, 'vec should be a vector'
    # Pielou eveness
    vec = vec / np.sum(vec)
    hm = np.log(np.sum(vec!=0))
    h = 0
    for i in range(len(vec)):
        if(vec[i]!=0):
            h += -1*vec[i]*np.log(vec[i])
    return h , hm

@jit(nopython=True)
def even2(vec):
    assert len(vec.shape)==1, 'vec should be a vector'
    # Heip evenness
    vec = vec / np.sum(vec)
    h = 0
    for i in range(len(vec)):
        if(vec[i]!=0):
            h += -1*vec[i]*np.log(vec[i])
    N1 = np.e**(h) - 1
    N2 = np.sum(vec!=0) - 1
    return N1 , N2

@jit(nopython=True)
def even12(vec):
    assert len(vec.shape)==1, 'vec should be a vector'
    # adaptation of Pielou eveness
    vec = vec / np.sum(vec)
    hm = np.log(len(vec))
    h = 0
    for i in range(len(vec)):
        if(vec[i]!=0):
            h += -1*vec[i]*np.log(vec[i])
    return h , hm

@jit(nopython=True)
def shannon(vec):
    # shannon biodiversity
    assert len(vec.shape)==1, 'vec should be a vector'
    vec = vec / np.sum(vec)
    h = 0
    for i in range(len(vec)):
        if(vec[i]!=0):
            h += -1*vec[i]*np.log(vec[i])
    return h

#%%
if(__name__=='__main__'):
    sp = 250 # the sinking speed (m/day)
    # The extend of the dataset used: 'glob' is global, 'SO' is Southern Ocean
    extend = 'glob'#'SO'
    if(extend=='SO'):
        maxlat = 5 # maximum latitude
    else:
        maxlat = 65 # maximum latitude
    # The biodiversity index used:
    biot = 'shannon'#'Heip'#'even' # 'rich'#
    
    # the v_{min} parameters used:
    minss = [100, 200, 300, 400, 500, 600,700,800,900,1000]
    # The \xi parameters used:
    xiss = np.arange(0.0001,0.012, 0.0001)
    
    # To keep track of the results for different parameters:
    Dbioa = np.zeros((len(minss), len(xiss))) # dinocysts
    Fbioa = np.zeros((len(minss), len(xiss))) # foraminifera
    
    for mini,mins in enumerate(minss): # loop over vmin values
        print('min:  %d'%(mins)) # to keep track of the progression
        for xii, xis in enumerate(xiss): # loop over xi values
            opts = ["xi", xis] # defines the xi parameters
            
            #%% Load the species data
            # Read Foram data
            readData = '/Volumes/HD/network_clustering/'
            # data matrix, contains relative abundances of species at the 
            # sediment sample locations
            data = readForamset(readData + 'ForamData.csv')
            Foramspecies = readForamset(readData + 'ForamDataHeader.txt')[0][21:]
            data[data=='N/A'] = '0'#'NaN'
            data = data[:, 21:].astype(np.float)
            # longitudes and latitudes of the sediment sample locations
            Flats, Flons = readForamset_lonlats(readData + 'ForamData.csv')
            Flons[np.where(Flons<0)]+=360
            
            # Read Dino data
            # longitudes and latitudes:
            FlonsDino, FlatsDino, Dinodata = readDinoset(readData+'dinodata_red.csv')
            FlonsDino[FlonsDino<0] += 360
            # data matrix of the dinocysts:
            Dinodata[np.isnan(Dinodata)] = 0

            # Reduce the foraminifera dataset to South of everyting of 
            # 'maxlat' degrees North
            idx = np.where(Flats<maxlat)[0]
            Flons = Flons[idx]
            Flats = Flats[idx]
            data = data[idx]
            # calculate the biodiversity. 'biot' is the type of biodiversity 
            # index used
            if(biot=='rich'): # species richness
                Fbio = np.sum((data>0), axis=1)
            elif(biot in ['even','shannon','Heip']):
                Fbio = biodiversityf(data, biot)
                
            # Reduce the dinocyst dataset to South of everyting of 
            # 'maxlat' degrees North
            idx = np.where(FlatsDino<maxlat)[0]
            FlonsDino = FlonsDino[idx]
            FlatsDino = FlatsDino[idx]
            Dinodata = Dinodata[idx]
            # calculate the biodiversity. 'biot' is the type of biodiversity 
            # index used
            if(biot=='rich'): # species richness
                Dbio = np.sum((Dinodata>0), axis=1)
            elif(biot in ['even','shannon','Heip']):
                Dbio = biodiversityf(Dinodata, biot)
    
            #%% Load the clustering result
            dirr = '/Users/nooteboom/Documents/GitHub/cluster_TM/cluster_SP/density/dens/results/'
            ff = np.load(dirr+'OPTICS_sp%d_smin%d.npz'%(sp, mins))
            lon0 = ff['lon']
            lat0 = ff['lat']
            reach = ff['reachability']
            ordering = ff['ordering']
            predecessor = ff['predecessor']
            core_distances = ff['core_distances']
            
            m, c = opts[0], opts[1]
            if m == "xi":
                l, _ = cluster_optics_xi(reach, predecessor, ordering, mins, xi=c)
            else:
                l = cluster_optics_dbscan(reachability=reach,
                                                    core_distances=core_distances,
                                                   ordering=ordering, eps=c)
            labels = np.array([li for li in l]) # labels of the clusters
            
            # find the closest release location to the
            # foraminifera sample locations, and cluster the sample location
            # accordingly
            args = find_nearest_args(lon0, lat0, Flats, Flons)
            Flabels = labels[args]
            # take the mean of the biodiversity outside and inside clusters
            biod_nc = np.nanmean(Fbio[Flabels==-1])
            biod_c = np.nanmean(Fbio[Flabels!=-1])
            
            # subtract the two to obtain the 'final'result
            Fbioa[mini,xii] =  biod_nc - biod_c
            
            # find the closest release location to the
            # dinocyst sample locations, and cluster the sample location
            # accordingly
            args = find_nearest_args(lon0, lat0, FlatsDino, FlonsDino)
            Dinolabels = labels[args]
            biod_nc = np.nanmean(Dbio[Dinolabels==-1])
            biod_c = np.nanmean(Dbio[Dinolabels!=-1])
            
            # subtract the two to obtain the 'final'result
            Dbioa[mini,xii] =  biod_nc - biod_c
    #%% save the results
    np.savez('cbG_shannon_%d'%(sp), xiss=xiss, smins = minss,
                     Dbiod=Dbioa, Fbiod=Fbioa)