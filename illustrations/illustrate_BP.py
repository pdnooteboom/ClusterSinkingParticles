#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:23:31 2020

@author: nooteboom
"""

import numpy as np
from matplotlib.pylab import plt
import seaborn as sns
import ot
import ot.plot
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def find_nearest_index(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def create_Bgraph():
    B = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    B.add_nodes_from([1, 2, 3, 4], bipartite=0)
    B.add_nodes_from(["a", "b", "c"], bipartite=1)
    # Add edges only between nodes of opposite node sets
    B.add_edges_from([(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "c"), 
                      (4, "a"), (4, "b")])
    
    return B

sns.set(context='paper',style="whitegrid",font="Arial", font_scale=1.2)

surfacecolor = 'dodgerblue'
firstcloudcolor = 'k'
secondcloudcolor = 'forestgreen'

#%%
fs = 22

xL = -30; yL = -30;
sigma = 9
sigma2 = 8
bias = 10

res = 3

con = 3
con2 = 32

n = 8

np.random.seed(1)

x1 = np.random.normal(xL+bias,sigma2,n) + 12*con
x2 = np.random.normal(xL,sigma,n)+14 

y1 = np.random.normal(yL,sigma2+2,n) + 16
y2 = np.random.normal(yL+bias,sigma,n)+con2 

#Define OT
M = ot.dist(np.concatenate((x1[:,np.newaxis],y1[:,np.newaxis]), axis=1), np.concatenate((x2[:,np.newaxis],y2[:,np.newaxis]), axis=1))
M /= M.max()

G0 = ot.emd(np.ones((n,)) / n, np.ones((n,)) / n, M)

sns.set_style("dark")

#%%
fig = plt.figure(figsize=(17,8))
gs = fig.add_gridspec(24, 6)
#%% transportation matrix
import matplotlib.image as mpimg


ax = fig.add_subplot(gs[:, :3])
img = mpimg.imread('TM.png')
ax.imshow(img)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_title('(a) Transportation matrix', fontsize=fs)
#%%
import networkx as nx
from networkx import bipartite

B = create_Bgraph()
X, Y = bipartite.sets(B)
pos = dict()
pos.update( (n, (i, 1)) for i, n in enumerate(X) ) # put nodes from X at x=1
pos.update( (n, (i-0.5, 2)) for i, n in enumerate(Y) ) # put nodes from Y at x=2

ax = fig.add_subplot(gs[:12, 3:-1])
ax.text(3.35,1.9,'surface', fontsize=fs)
ax.text(3.35,1.05,'bottom', fontsize=fs)
nx.draw(B, pos=pos, ax=ax, node_color="k",
        width=2)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_title('(b) Bipartite graph', fontsize=fs)

G = bipartite.projected_graph(B, [1, 2, 3, 4], multigraph=True)
ax = fig.add_subplot(gs[13:, 3:-1])
#ax.set_title('\n bottom projection', fontsize=fs)
ax.text(0.6,3.03,'(c) Bottom projection', fontdict={'size':fs, 'color':'k'})
pos = dict()
pos.update( (n, (i, 3)) for i, n in enumerate(X) ) # put nodes from X at x=1
nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color="k")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_facecolor('white')

edges = list(G.edges)
weights = {}
edg = []
for l in range(len(edges)):
    ed = str(edges[l][0]) + str(edges[l][1])
    if(ed in weights.keys()):
        weights[ed] += 1
    else:
        weights[ed] = 1
        edg.append([edges[l][0],edges[l][1]])

import matplotlib.patches as patches
style = "Simple, tail_width=2, head_width=0, head_length=1"
kw = dict(arrowstyle=style, color="k")
for l in range(len(edges)):
#    psy = (pos[edges[l][1]][0], 0.01+pos[edges[l][1]][1])
    a = patches.FancyArrowPatch(pos[edges[l][0]], pos[edges[l][1]],#psy,
                             connectionstyle="arc3,rad=.43",
                             **kw)
    ax.add_patch(a)
    
for l in range(len(edg)):
    we = str(edg[l][0]) + str(edg[l][1])
    if(we=='14'):
        ytext = pos[edges[l][0]][1]-0.04
        xtext = (edg[l][0] + edg[l][1]) / 2 -1
    elif(we=='24'):
        ytext = pos[edges[l][0]][1]-0.015
        xtext = (edg[l][0] + edg[l][1]) / 2 -1
    else:
        ytext = pos[edges[l][0]][1]-+0.005
        xtext = (edg[l][0] + edg[l][1]) / 2 -1.1#pos[edges[l][0]][1] +0.5#- pos[edges[l][0]][1]
    print(xtext,ytext)
    plt.text(xtext, ytext, str(weights[we]), fontdict={'size':fs, 'color':'k'})

plt.savefig('illustrationBP.png', bbox_inches='tight', dpi=300)
plt.show()
