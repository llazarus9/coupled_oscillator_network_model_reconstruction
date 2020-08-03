# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:22:47 2020

@author: Lauren
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy import optimize
from functools import partial
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import warnings
from sklearn.metrics import roc_curve, auc, f1_score
from scipy import interpolate
from scipy.sparse import csr_matrix, lil_matrix,vstack,hstack
from scipy.sparse.linalg import spsolve
import scipy.signal as sp


class Network():
    def __init__(self,system_params,solution_params):
        
        for key, value in system_params.items():
            setattr(self, key, value)
        
        self.solution_params = solution_params
        
        
        
    def gen_erdos_renyi(self):
        '''
        Generates A adjacency matrix for random network 
        given E-R probability and number of oscs.
        '''
        
        if int(self.seed)>0:
            np.random.seed(int(self.seed))
            
        A = np.matrix(np.random.choice(a=[0,1],
                                 p=[1-np.sqrt(self.p_value),np.sqrt(self.p_value)],
                                 size=(self.num_osc,self.num_osc)))
        A = A-np.diag(np.diag(A))
        self.A = np.multiply(A,A.T)
        
        
        
    def gen_frequencies(self):
        '''
        Generates random frequencies for each oscillator.
        '''
        
        if int(self.seed)>0:
            np.random.seed(int(self.seed))
            
        self.w = self.mu+self.sigma*np.random.randn(self.num_osc,1).astype('float32')
        
        
        
    def kuramoto_rhs(self,Gamma=self.Gamma):
        
        
    def gen_data(self):
        
        
        
class NetworkData():
    def __init__(self,phase):
        
        self.phase = phase
        self.num_oscs = phase.shape[1]
        self.n_timestep = phase.shape[0]
        
        self.vel = self.get_vel()
        
        
        
    def get_vel(self):
        '''
        central_diff
        '''
        
        
        
        
    def gen_diffs(self):
        '''
        get_diff_mat
        '''
        y = self.phase
        
        nrows=y.shape[0]
        ncols=y.shape[1]
        finaldiffmat=np.zeros(shape=(nrows,ncols,ncols))
        
        for index in range(nrows):
            row=y[index,:]
            rowvec=np.array(row,ndmin=2)
            colvec=np.transpose(rowvec)
            diffmat=-(rowvec-colvec)
            finaldiffmat[index,:,:]=diffmat
        
        self.diffs = finaldiffmat
        
        
    def gen_training_test_data(self,frac=0.8):
        
        self.gen_diffs
        
        n_timestep = self.diffs.shape[0]
        inds = np.random.permutation(n_timestep)
        stop = int(np.ceil(frac*n_timestep))
        traininds = inds[:stop]
        testinds = inds[stop:]
        
        phase_train = self.phase[traininds,:]
        diff_train = self.diffs[traininds,:,:]
        vel_train = self.vel[traininds,:]
        
        phase_test = self.phase[testinds,:]
        diff_test = self.diffs[testinds,:,:]
        vel_test = self.vel[testinds,:]
        
        return phase_train,diff_train,vel_train,phase_test,diff_test,vel_test
        