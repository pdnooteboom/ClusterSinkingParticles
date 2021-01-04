#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:15:35 2020

@author: nooteboom
"""
from numba import jit
import numpy as np
import math
import matplotlib.pylab as plt
import matplotlib
from scipy import sparse
import networkx as nx
import scipy
from pandas import read_csv
from time import time

assert __name__!='__main__'

#%% To read datasets
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


def readDinoset(name):
    lons = read_csv(name, usecols=[0]).values[:,0]
    lats = read_csv(name, usecols=[1]).values[:,0]
    data = read_csv(name).values[:,2:]
    data[np.isnan(data)] = 0
    row_sum = data.sum(axis=1)
    data = data / row_sum[:,np.newaxis]
    
    return lons, lats, data
#%% To calculate similarity measures

#@jit(nopython=True)
def Euclidean_distance(x,y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    if(res==0):
        res = 1 / math.sqrt(res / 4)
    return 1 / math.sqrt(res / 4)

@jit(nopython=True)
def Cos_sim(x,y):
    res = 0
    normx = 0
    normy = 0
    for i in range(len(x)):
        res += (x[i]* y[i])
        normx += x[i]**2
        normy += y[i]**2
    return ((res / math.sqrt(normx*normy)))#/2

@jit(nopython=True)
def Similarity_data(data):
    res = np.ones((data.shape[0],data.shape[0]))
    for i in range(res.shape[0]):
        for j in range(i):
             EC = Cos_sim(data[i],data[j])
             res[i,j] = EC
             res[j,i] = res[i,j]
    return res

#%%

@jit(nopython=True)
def get_edges(TM):
    res = []
    for i in range(TM.shape[0]):
        for j in range(TM.shape[1]):
            if(TM[i,j]>0):
                res.append((i,-j))
    return res

@jit(nopython=True)
def get_weighted_edges(TM):
    res = []
    for i in range(TM.shape[0]):
        for j in range(TM.shape[1]):
            res.append((i,-j, TM[i,j]))
    return res

#%% classes 
class bipartite_network(object):
    """
    Class to  handle bipartite network analysis.    
    """
    
    def __init__(self, B):
        self.adjacency_matrix = B
        self.N = B.shape[0]
        self.M = B.shape[1]        


    def projection_adjacency_matrix(self, space = 'X'):
        """
        Return adjacency matrix for projection, i.e. GG^T (or G^TG)
        """
        if space == 'X':
            return self.adjacency_matrix.dot(self.adjacency_matrix.transpose())
        
        elif space == 'Y': 
            return self.adjacency_matrix.transpose().dot(self.adjacency_matrix)

     
    def projection_laplacian_spectrum(self, K=20):
        """
        Return spectrum of the projection of the bipartite network, cf. eq. (10)
        """
        
        p = np.array(sparse.csr_matrix.sum(self.adjacency_matrix.T, axis=1))[:,0]
        p = self.adjacency_matrix.dot(p)
        PI_p_inv_sqrt = sparse.diags([1/np.sqrt(pi) if pi!=0 else 0 for pi in p])
        R = PI_p_inv_sqrt.dot(self.adjacency_matrix)
        u, s, vt = sparse.linalg.svds(R, K)
        indices = np.argsort(s)[::-1]    
        u=u[:,indices]
        s = s[indices]
        vt = vt[indices,:]
        return [PI_p_inv_sqrt.dot(u), s, vt.transpose()]


def construct_dendrogram(networks):
    """
    Construct a dendrogram from a hierarchical list of 'undirected_network' objects
    - networks: list of network groups. len(networks[0])=1, len(networks[1])=2, etc.
    Note that this ia a bit hacky, as the NCut increases with the number of clusters, which is
    different from the usual dendrogram... Therefore, we flip the axis later when plotting.
    """
    K = len(networks)
    network_labels = [[nw.cluster_label for nw in networks[i]] for i in range(K)]
    linkage_labels = [[nw.cluster_label for nw in networks[i]] for i in range(K)]
    original_obs = [[1 for _ in networks[i]] for i in range(K)]
    n = len(network_labels[-1])
    Z = np.zeros((n-1,4)) #linkage function
    for i in range(1,len(network_labels))[::-1]:
        network_label1 = network_labels[i][-1]
        network_label2 = network_labels[i][-2]
        label = np.min([network_label1, network_label2])
        i_merged = np.argwhere(network_labels[i-1]==label)[0][0]
        new_label = n + n-1-i
        old_label = linkage_labels[i-1][i_merged]
        for j in range(i):
            linkage_labels[j] = [l if l!=old_label else new_label for l in linkage_labels[j]]
            original_obs
        Z[n-1-i][0] = linkage_labels[i][-1]
        Z[n-1-i][1] = linkage_labels[i][-2]
        
    ncut = []
    for i in range(len(networks)):
        r = np.sum([networks[i][j].rho for j in range(len(networks[i]))])
        ncut.append(len(networks[i])-r)
    
    ncut=np.array(ncut)
    #For the Ncut axis, we make an ugly hack (as the Ncut increases with the number of clusters).
    #First define the Ncuts the wrong way around. Then adjust it in the plotting script. 
    Z[:,2]= np.max(ncut[1:]) - ncut[1:][::-1]+.2
    return Z
    

class undirected_network(object):
    """
    Class to handle analysis of undirected networks
    """
    
    def __init__(self, adjacency_matrix, cluster_indices=np.array([]), 
                 cluster_volume=np.array([]), cluster_label = 0):
        
        """
        - adjacency_matrix: format sparse.csr_matrix. If it is not symmetric it is symmetrized.
        - cluster_indices: indices corresponding to network domain.
        - cluster_volume: vector of volume of the nodes inside the cluster. The volume of a node is equal to
        the sum over all the weights it connects to. The colume of a set of nodes is equal to the
        denomiator of a term in NCut, cf. eq. (6)
        - cluster_label: each cluster receives a label so that we can later distinguish them.
        """
        if(len(cluster_indices)==0): cluster_indices = np.array(range(adjacency_matrix.shape[0]))
        if(len(cluster_volume)==0): cluster_volume = np.array(sparse.csr_matrix.sum(adjacency_matrix, axis=1))[:,0]
        
        self.adjacency_matrix = adjacency_matrix
        self.cluster_indices = cluster_indices
        self.cluster_volume = cluster_volume
        self.N = adjacency_matrix.shape[0]
        self.cluster_label = cluster_label
        self.rho = np.sum(self.adjacency_matrix)/np.sum(self.cluster_volume)
        assert(len(cluster_indices) == self.adjacency_matrix.shape[0])
        assert(len(cluster_volume) == len(cluster_indices))
        print('Construct undirected network.')
    
    
    def __del__(self):
        print('Adjacency matrix object deleted')

      
    def largest_connected_component(self):

        """
        Determine connected components
        """
        
        print('Find largest connected component.')
        
        G = nx.from_scipy_sparse_matrix(self.adjacency_matrix, create_using = nx.Graph())
        components = np.array(list(nx.connected_components(G)))
        component_lengths = np.array([len(s) for s in components])
        component_inds = np.argsort(component_lengths)[::-1]
        components_sorted = components[component_inds]
        component_lengths = np.array([len(c) for c in components_sorted])
        
        print('Lengths of components (>1)')
        print(component_lengths[component_lengths>1])
        
        #Largest component
        inds = list(components_sorted[0])
        sub_adjacency_matrix = self.adjacency_matrix[inds, :][:, inds]
        sub_cluster_indices = self.cluster_indices[inds]
        sub_cluster_volume = self.cluster_volume[inds]
        sub_cluster_label = self.cluster_label
        return undirected_network(sub_adjacency_matrix, sub_cluster_indices, 
                                               sub_cluster_volume, sub_cluster_label)
        
    
    def compute_laplacian_spectrum(self, K=20, plot=False):
        """
        Comput eigenvectors for clustering from symmetric nocmralized Laplacian
        """
        d = np.array(sparse.csr_matrix.sum(self.adjacency_matrix, axis=1))[:,0]
        D_sqrt_inv = scipy.sparse.diags([1./np.sqrt(di) if di!=0 else 0 for di in d ])
        L = sparse.identity(self.N) - (D_sqrt_inv.dot(self.adjacency_matrix)).dot(D_sqrt_inv)
        print('Computing spectrum of symmetric normalized Laplacian')
        w, v = sparse.linalg.eigsh(L, k=K, which = 'SM')
        inds = np.argsort(w)
        w = w[inds]
        v = v[:,inds]
        
        if plot:
            plt.plot(w, 'o')
            plt.title('Eigenvalues of symmetric normalized Laplacian')
            plt.grid(True)
            plt.show()
        
        self.Lsym_eigenvectors =  D_sqrt_inv.dot(v)
        self.Lsym_eigenvalues =  D_sqrt_inv.dot(v)
        return w, D_sqrt_inv.dot(v)
   

    def drho_split(self, indices_1, indices_2):
        """
        If we propose to split a cluster, this function returns the changes in the coherence ratio for a split into
        indices_1 and indices_2. We maximize this change in the coherence ratio, which is equal to
        minimizing the NCut.
        """
        cluster_volume_1 = np.sum(self.cluster_volume[indices_1])
        cluster_volume_2 = np.sum(self.cluster_volume[indices_2])
        stays_in_1 = np.sum(self.adjacency_matrix[indices_1, :][: ,indices_1])
        stays_in_2 = np.sum(self.adjacency_matrix[indices_2, :][: ,indices_2])        
        return stays_in_1 / cluster_volume_1 + stays_in_2 / cluster_volume_2 - self.rho


    def hierarchical_clustering_ShiMalik(self, K, plots=False):
        """
        Implementation of hierarchical clustering according to Shi & Malik 2000.
        At each iteration, one cluster is added, minimizing the global NCut. We implement this
        by computing the increase in the coherence ratio rho and choose the maximum increase. This is
        equivalent to minimizing the global NCut, cf. eq. A2
        """
        networks = {}
        networks[0] = [self]
        boolean = True
        i = 0
        while(i<K and boolean):#for i in range(1,K):
            i += 1
            print('Level: ', i)
            
            optimal_drhos = []
            optimal_cutoffs = []
            
            for j in range(len(networks[i-1])):
                nw = networks[i-1][j]
                if nw.N<100: 
                    optimal_drhos.append(np.nan)
                    optimal_cutoffs.append(np.nan)
                    continue
        
                nw.compute_laplacian_spectrum()
                V_fiedler = nw.Lsym_eigenvectors[:,1]
                c_range = np.linspace(np.min(V_fiedler), np.max(V_fiedler), 100)[1:]
                
                drhos = []
                for c in c_range:
                
                    indices_1 = np.argwhere(V_fiedler<=c)[:,0]
                    indices_2 = np.argwhere(V_fiedler>c)[:,0]
                    drhos.append(nw.drho_split(indices_1, indices_2))
                    
                drhos = np.array(drhos)
                if plots:
                    plt.plot(c_range, drhos)
                    plt.yscale('log')
                    plt.grid(True)
                    plt.title(r'$\Delta \rho_{global}$ for different cutoffs. Network' + str(i) + str(j))
                    plt.show()
                cutoff_opt = c_range[np.nanargmax(drhos)]
                print('Choosing as cutoff: ', str(cutoff_opt))
                
                optimal_drhos.append(np.nanmax(drhos))
                optimal_cutoffs.append(cutoff_opt)
            
            if(np.isnan(optimal_drhos).all()):
                boolean = False
            else:
                i_cluster = np.nanargmax(optimal_drhos)
                print('Splitting cluster ', i_cluster+1)
                cutoff_cluster = optimal_cutoffs[np.nanargmax(optimal_drhos)]
                nw_to_split = networks[i-1][i_cluster]
                V_fiedler = nw_to_split.Lsym_eigenvectors[:,1]
                indices_1 = np.argwhere(V_fiedler<=cutoff_cluster)[:,0]
                indices_2 = np.argwhere(V_fiedler>cutoff_cluster)[:,0]
                
                #If a cluster is split, the largest sub-cluster receives the same label.
                if len(indices_1)<len(indices_2):
                    ind_ = indices_1.copy()
                    indices_1= indices_2.copy()
                    indices_2 = ind_
                
                adjacency_matrix_1 = nw_to_split.adjacency_matrix[indices_1, :][:, indices_1]
                adjacency_matrix_2 = nw_to_split.adjacency_matrix[indices_2, :][:, indices_2]
                cluster_indices_1 = nw_to_split.cluster_indices[indices_1]
                cluster_indices_2 = nw_to_split.cluster_indices[indices_2]
                cluster_volume_1 = nw_to_split.cluster_volume[indices_1]
                cluster_volume_2 = nw_to_split.cluster_volume[indices_2]
                
                cluster_label_1 = nw_to_split.cluster_label
                
                old_labels = [nw.cluster_label for nw in networks[i-1]]
                
                cluster_label_2 = np.max(old_labels)+1
                
                network_children = [undirected_network(adjacency_matrix_1, cluster_indices_1, cluster_volume_1, cluster_label_1), 
                                undirected_network(adjacency_matrix_2, cluster_indices_2, cluster_volume_2, cluster_label_2)]
                
                networks[i] = networks[i-1].copy()
                networks[i].pop(i_cluster)
                networks[i] += network_children #append in the end
            
        self.clustered_networks = networks
        
class Geographic_bounds(object):
    """
    Class to handle analysis of boundaries between clusters (figure 2b in 
    manuscript)
    """
    
    def __init__(self, dat, networks):
        
        """
        - 
        """
        self.cmap = matplotlib.cm.get_cmap('inferno')
        self.colors = []
        self.oldclus = []
        self.lons = []
        self.lats = []
        self.ccs = []
        self.directions = []
        self.colors = []
        self.dat = dat
        self.networks = networks
        
    def create(self,its):
        vLats = self.dat['vLats']; vLons= self.dat['vLons'];
        for c in range(its+1):
            ti = time()
            self.colors.append(self.cmap(c / its))
            field_plot = np.ones(self.dat['field_plot'].shape)*(-10000)
            for k in range(c): 
                field_plot[self.networks[c-1][k].cluster_indices]= self.networks[c-1][k].cluster_label

            lo, la, cu, oc, direc = self.get_bounds_from_clusters(self.oldclus,
                                                             field_plot, 
                                                             vLons, 
                                                             vLats)
            self.lons.append(lo); self.lats.append(la); self.ccs.append(cu); 
            self.oldclus.append(oc); self.directions.append(direc)
            print('iteration took %.1f minutes'%((time()-ti)/60))
        
    def id_lonlat(self,vLons, vLats, lon, lat):
        return np.where(np.logical_and(vLons==lon, vLats==lat))[0][0]
    
    def loc_not_in_array(self,vLons, vLats, lon, lat):
        return ~(np.logical_and(vLons==lon, vLats==lat).sum()>=1)
    
    def dist(self,lo1, lo2, la1, la2):
        return np.sqrt((lo1-lo2)**2+(la1-la2)**2)
        
    def set_bounds(self, lons, lats, fieldplot, idx):
        reslon = []
        reslat = []
        resdist = []
        for i in range(len(lons)):
            print('newlon %d \n'%(i))
            if(len(reslon)==0):
                lon = lons[i]
                lat = lats[i]
                reslon.append(lon)
                reslat.append(lat)
            else:
                lon = reslon[-1]
                lat = reslat[-1]
                bo = True
                
                k = 0
                while(bo):
                    if(k>=len(lons)):
                        bo = False
                    else:
                        dd = self.dist(lon, lons[k], lat, lats[k])
                        if(dd<=np.sqrt(0.5)+0.01 and dd!=0):
                            if(self.loc_not_in_array(reslon, reslat, lons[k], lats[k])):
                                reslon.append(lons[k])
                                reslat.append(lats[k])
                                resdist.append(dd)
                                bo = False
                    k += 1
                            
        # make the boundary periodic
        reslon.append(reslon[0])
        reslat.append(reslat[0])
        return reslon, reslat
      
    def to_compile(self, newclus, fieldplot, idx, vLons, vLats):
        mlo = max(vLons)
        boundlons = []
        boundlats = []
        leftneighbours = np.zeros(len(idx))
        rightneighbours = np.zeros(len(idx))
        downneighbours = np.zeros(len(idx))
        upneighbours = np.zeros(len(idx))
        for i in range(len(idx)):
            lon = vLons[idx[i]]
            lat = vLats[idx[i]]
            boundlons.append(lon)
            boundlats.append(lat)
            if(self.loc_not_in_array(vLons[idx], vLats[idx], (lon-1)%mlo, lat)):
                leftneighbours[i] = 1
            if(self.loc_not_in_array(vLons[idx], vLats[idx], (lon+1)%mlo, lat)):
                rightneighbours[i] = 1
            if(lat!= np.min(vLats) and self.loc_not_in_array(vLons[idx], vLats[idx], 
               (lon)%mlo, lat-1)):
                downneighbours[i] = 1
            if(lat != np.max(vLats) and self.loc_not_in_array(vLons[idx], vLats[idx], 
               (lon)%mlo, lat+1)):
                upneighbours[i] = 1
    
        leftlons = np.array(boundlons)[leftneighbours==1] - 0.5
        leftlats = np.array(boundlats)[leftneighbours==1]
        rightlons = np.array(boundlons)[rightneighbours==1] + 0.5 
        rightlats = np.array(boundlats)[rightneighbours==1]  
        downlons = np.array(boundlons)[downneighbours==1]  
        downlats = np.array(boundlats)[downneighbours==1] - 0.5  
        uplons = np.array(boundlons)[upneighbours==1]  
        uplats = np.array(boundlats)[upneighbours==1] + 0.5    
    
        return leftlons, leftlats, rightlons, rightlats, downlons, downlats, uplons, uplats
    
    def check_neighbours(self, newclus, fieldplot, idx, vLons, vLats):
        leftlons, leftlats, rightlons, rightlats, downlons, downlats, uplons, uplats = self.to_compile(newclus, fieldplot, idx, vLons, vLats)
    
        lons = np.concatenate((leftlons,rightlons,uplons,downlons))
        lats = np.concatenate((leftlats, rightlats, uplats, downlats))
        direction = np.concatenate((np.full(len(leftlons), 0), np.full(len(rightlons),0),
                                            np.full(len(uplons),1),np.full(len(downlons),1)))
        return lons, lats, direction
        
    
    def not_intersection(self,lst1, lst2): 
        lst3 = [value for value in lst1 if value not in lst2] 
        return lst3
    
    def get_bounds_from_clusters(self, oldclus, fieldplot, vLons, vLats):
        newclus = np.setdiff1d(np.unique(fieldplot), oldclus, assume_unique=True)[-1]
        idx = np.where(fieldplot==newclus)[0]
        lons, lats, direc = self.check_neighbours(newclus, fieldplot, idx, vLons, vLats)
        clusters = np.full(len(lons), newclus)
        return lons, lats, clusters, newclus, direc
    
    def plot_bounds(self,lo, la, direc, c='k'):
        # function that is used for plotting
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
            plt.plot(plo[l], pla[l], c=c)
