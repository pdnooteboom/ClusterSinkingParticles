#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:10:39 2020

@author: nooteboom
"""
import numpy as np
import matplotlib.pylab as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


def CCAplot(CA, Y, envs, clus=None, dataset=''):
    fs = 15
    fig, ax = plt.subplots(figsize=(10,10))#plt.figure(figsize=(10,10))
    ax.set_title(dataset, fontsize=fs)
    ax.set_xlabel('first axis', fontsize=fs)
    ax.set_ylabel('second axis', fontsize=fs)
    
    for f in range(len(CA.features['CCA1'])):
        ax.arrow(0,0,CA.features['CCA1'][f],CA.features['CCA2'][f],
                  head_width=0.08, color='k')
        ax.annotate(envs[f], (CA.features['CCA1'][f],-0.15+CA.features['CCA2'][f]))
    if(clus is not None):
        ax.scatter(CA.samples['CCA1'],CA.samples['CCA2'], c=clus, 
                   label='scores', cmap='tab10')        
    else:
        ax.scatter(CA.samples['CCA1'],CA.samples['CCA2'], label='scores')
    
    plt.show() 

def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

def calculate_rsquared(model, X, y):
    yhat = model.predict(X)
    SS_Residual = sum((y-yhat)**2)       
    SS_Total = sum((y-np.mean(y))**2)     
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
    return adjusted_r_squared#r_squared#, adjusted_r_squared

def regres(inte, coef, val):
    return inte + coef*val

def plot_regres(ax,reg, X):
    inte=reg.params[0]
    coef = reg.params[1]
    ax.plot([np.min(X), np.max(X)], 
              [regres(inte, coef, np.min(X)),
               regres(inte, coef, np.max(X))], c='k')

def square_table(tabel):
    l = 0
    for i in range(len(tabel)):
        if(l<len(tabel[i])):
            l = len(tabel[i])
    for i in range(len(tabel)):
        for j in range(l-len(tabel[i])):
            tabel[i].append('')
    return tabel