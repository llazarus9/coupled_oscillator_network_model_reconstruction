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
    def __init__(self,system_params):
        
        for key, value in system_params.items():
            setattr(self, key, value)
        
        self.gen_erdos_renyi()
        self.gen_frequencies()
        
        
    def add_ivp_info(self,solution_params):
        
        for key, value in solution_params.items():
            setattr(self, key, value)
        
        if isinstance(self.IC,dict):
            self.perturbs = True
        else:
            self.perturbs = False
        
        
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
        
        
    def dydt_kuramoto(self,t,y):
        y = np.reshape(y,(-1,1))    # makes y a column vector
        dydt = self.w + self.K*np.mean(np.multiply(self.A,self.Gamma(y.T-y)),axis=1)
        
        return dydt
        
    
    def gen_data(self):
        # bulk here missing
        return phases_all
        
        
class NetworkData():
    def __init__(self,phase,dt):
        
        self.phase = phase
        self.num_oscs = phase.shape[1]
        self.n_timestep = phase.shape[0]
        
        self.dt = dt
        self.get_vel(with_filter=True,truncate=False,return_phases=True)
        
        
        
    def get_vel(self,with_filter=True,truncate=True,return_phases=True):
        '''
        See: central_diff.  Value dt assumed constant.

        Parameters
        ----------
        with_filter : Boolean, optional
            Use Savitsky-Golay filter? The default is True.
        truncate : Boolean, optional
            Only keep internal time points? The default is True.
        return_phases : Boolean, optional
            Return both velocities and phases? The default is True.
        '''

        
        # dt=t[2:]-t[:-2]
        dy = self.phase[2:,:]-self.phase[:-2,:]
        deriv = dy/(2*self.dt)   # first approximation (y_{i+1}-y{i-1})/(2*dt)
        
        if with_filter:        
            deriv = sp.savgol_filter(deriv, 5, 1, axis=0)
            
        if truncate:
            phases = self.phase[1:-1,:]
        else:
            phases = self.phase
            deriv = np.concatenate([np.nan*np.zeros(shape=(1,self.num_osc)),deriv,
                                    np.nan*np.zeros(shape=(1,self.num_osc))])
            
        if return_phases:
            self.phase = phases
        self.vel = deriv
        
        
    def gen_diffs(self):
        '''
        See: get_diff_mat.
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
        '''
        See: get_training_testing_data and get_split.

        Parameters
        ----------
        frac : TYPE, optional
            Proportion of data to use as training data. The default is 0.8.

        Returns
        -------
        phase_train : TYPE
            DESCRIPTION.
        diff_train : TYPE
            DESCRIPTION.
        vel_train : TYPE
            DESCRIPTION.
        phase_test : TYPE
            DESCRIPTION.
        diff_test : TYPE
            DESCRIPTION.
        vel_test : TYPE
            DESCRIPTION.

        '''
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
        