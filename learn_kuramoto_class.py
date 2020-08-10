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
import sys


class KuraRepsNetwork():
    def __init__(self,system_params):
        
        for key, value in system_params.items():
            setattr(self, key, value)
        
        self.gen_erdos_renyi()
        self.gen_frequencies()
        
        
    def add_ivp_info(self,solution_params):
        
        for key, value in solution_params.items():
            setattr(self, key, value)
        
        if any(self.IC) and isinstance(self.IC,dict):    # IC = IC_setup
            self.perturbs = True
        else:                           # IC = {} or IC = given
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
        
    
    def solve_kuramoto_ode(self):
        
        tmin = 0.0
        numsteps = int(np.round((self.tmax-tmin)/self.dt)) # number of steps to take
        t_eval = np.linspace(tmin,self.tmax,numsteps+1)
        
        if self.dyn_noise > 0:
            t,y = solve_ivp_stochastic_rk2(self.dydt_kuramoto,
                                 t_eval.reshape(-1,1),
                                 self.curr_IC.reshape(1,-1),self.dyn_noise)
        else:
            sol = solve_ivp(self.dydt_kuramoto, 
                          (tmin,self.tmax),
                          self.curr_IC,
                          method='RK45',
                          t_eval=t_eval,
                          vectorized=True)  # consider smaller rtol(def=1e-3)
            t=(np.reshape(sol.t,(-1,1)))
            y=(sol.y.T)
            
        return t,y
    
    
    def gen_perturb(self,y):
        
        pert=np.zeros(self.num_osc)
        
        # which oscillator(s) to perturb
        if self.IC['selection'] == 'fixed':
            inds = self.IC['indices']
        else:   # random
            inds = np.random.choice(range(self.num_osc),
                                    size=self.IC['num2perturb'],replace=False)
        
        ## type of perturbation
        if self.IC['type'] == 'reset':
            pert[inds] = -y[inds]
        else:   # random
            pert[inds] += np.random.randn(len(inds))*self.IC['size']
        
        return pert.reshape(1,-1)
    
    
    def solve(self):
        
        for k in range(self.num_repeats):
            
            if self.perturbs and (k > 0):   # perturbation from previous
                lasty = y[-1,:]
                lasty = lasty - int(lasty.mean()/(2*np.pi))*2*np.pi
                
                pert = self.gen_perturb(lasty)
                print("Repeat {}, phase perturbation: {}".format(k,pert.squeeze()))
                self.curr_IC = lasty.squeeze() + pert.squeeze()
            
            elif self.perturbs:             # starting IC for a perturb set
                self.curr_IC = self.IC['IC']
                
            elif any(self.IC):              # same IC all restarts
                self.curr_IC = self.IC
            else:                           # no IC given - re-randomize
                self.curr_IC = np.array(2.0*np.pi*np.random.rand(self.num_osc))
            
            t,y = self.solve_kuramoto_ode()
            y = y + self.noise*np.random.randn(y.shape[0],y.shape[1])
            
            deriv,phases = central_diff(t,y,with_filter=True,truncate=False,return_phases=True)
            
            n_ts = len(t)-2
            t = t[range(1,n_ts,self.ts_skip)]
            
            if k==0:
                phases_all = y[range(1,n_ts+1,self.ts_skip),:]
                deriv_all = deriv[range(1,n_ts+1,self.ts_skip),:]
            else:
                phases_all = np.vstack((phases_all,y[range(1,n_ts+1,self.ts_skip),:]))
                deriv_all = np.vstack((deriv_all,deriv[range(1,n_ts+1,self.ts_skip),:]))
        
        return phases_all,deriv_all
        

def solve_ivp_stochastic_rk2(dydt,T,IC,D):
    ''' 
    solve_ivp_stochastic_rk2(dydt,T,IC,D): 
        Solve the stochastic ode given by dy=dydt(y) dt+D sqrt(dt)
    
    Inputs:
    IC: numpy matrix (1,num_osc)
    dydt: function
    T: numpy array of times
    D: scalar (noise level)
       
    Outputs:  
    T: numpy array of times
    f(T.squeeze()): matrix of phases (numsteps,num_oscillators)
    ''' 
    # Uses RK2 (improved Euler) with gaussian white noise
    # D is the noise level
    dt = 0.05
    t = np.arange(T.min(),T.max()+dt,dt).reshape(-1,1)
    f = lambda y: dydt(0,y)
    Y = np.zeros((len(t),IC.shape[1]))
    oldy = IC
    Y[0,:] = oldy
    for k in range(len(t)-1):
        newy = rkstep(f,oldy,dt,D)
        Y[k+1,:] = newy.squeeze()
        oldy = newy
    f = interpolate.interp1d(t.squeeze(), Y,axis=0)   
    return T,f(T.squeeze())


def rkstep(f,y,dt,D):
    ''' 
    rkstep(f,y,dt,D): 
        Compute a single step for the stochastic ode described by dy=f(y)dt+D sqrt(dt)
    
    Inputs:
    y: numpy matrix (1,num_osc)
    f: function
    dt: scalar (timestep)
    D: scalar (noise level)
       
    Outputs:  
    newy: numpy matrix (1,num_osc)
    ''' 
    # See  Honeycutt - Stochastic Runge-Kutta Algorithms I White Noise - 1992 -
    # Phys Rev A
    psi = np.random.randn(y.shape[1]);
    k1 = f(y.T).T
    k2 = f((y+dt*k1+np.sqrt(2*D*dt)*psi).T).T
    newy = y + dt/2*(k1+k2) + np.sqrt(2*D*dt)*psi
    return newy


def central_diff(t,y,with_filter=True,truncate=True,return_phases=True):
    '''
    central_diff(t,y,with_filter,truncate):
        estimate derivative
    
    Inputs:
    t: time vector (n,1)
    y: phase vector (n,m)
    with_filter: boolean
    truncate: boolean
    return_phases: boolean
    
    Outputs:
    phases: matrix with phases at timestep i (num_timesteps,num_osc)
    deriv: matrix with velocities at timestep i  (num_timesteps,num_osc)
      
    '''
    num_osc = y.shape[1]
    dt = t[2:]-t[:-2]
    dy = y[2:,:]-y[:-2,:]
    deriv = dy/dt
    
    if with_filter:        
        deriv = sp.savgol_filter(deriv, 5, 1,axis=0)
    
    if truncate:
        phases = y[1:-1,:]
    else:
        phases = y
        deriv = np.concatenate([np.nan*np.zeros(shape=(1,num_osc)),deriv,
                                np.nan*np.zeros(shape=(1,num_osc))])

    if return_phases:
        return deriv,phases
    else:
        return deriv


def get_op(y):
    ''' 
    get_op(y): 
        Compute order parameter from phases
    
    Inputs:
    y: numpy matrix (num_timesteps,num_osc)
       
    Outputs:  
    R: magnitude of order parameter (num_timesteps,1)
    Psi: angle of order parameter (num_timesteps,1)
    ''' 
    Z = np.mean(np.exp(1j*y),axis=1)
    R = np.abs(Z)
    Psi = np.angle(Z)
    return R,Psi



class NetworkData():
    def __init__(self,phase,vel=None,t=None,dt=None):
        
        self.phase = phase
        self.num_oscs = phase.shape[1]
        self.n_timestep = phase.shape[0]
        
        if (t is None) and (dt is None) and (vel is None):
            sys.exit('Require vel, t, or dt for velocity computation.')
            
        elif (vel is not None):
            self.vel = vel
            
        else:
            if (t is not None):
                self.t = t
            else:
                self.t = [(0+i*dt) for i in range(self.n_timestep)]
                
            self.vel = central_diff(self.t,self.phase,with_filter=True,
                                    truncate=False,return_phases=False)
        
        
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
        

