# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 09:52:07 2020

@author: Lauren
"""

import learn_kuramoto_class as lk
import numpy as np
import importlib as imp
import matplotlib.pyplot as plt
imp.reload(lk)

import warnings
warnings.filterwarnings("ignore")

##############################################################################
## define model parameters

num_osc = 10
mu_freq = 0.0         # mean natural frequency
sigma_freq = 0.01     # std natural frequency
p_erdos_renyi = 0.9   # probability of connection for erdos renyi
random_seed = -1      # -1 to ignore
coupling_function = lambda x: np.sin(x)   # coupling function

system_params = {'num_osc': num_osc,
                 'seed': random_seed,
                 'mu': mu_freq,
                 'sigma': sigma_freq,
                 'p_value': p_erdos_renyi,
                 'K': 1.0,
                 'Gamma': coupling_function
                 }

my_network = lk.Network(system_params)

##############################################################################
## define numerical solution parameters

dt = 0.1            # time step for numerical solution
tmax = 1000*dt      # maximum time for numerical solution
noise_level = 0.0           # observation noise added
dynamic_noise_level = 0.00  # oscillator dynamics noise added

num_repeats = 10            # number of restarts for numerical solution
IC_setup = {'type': 'reset',          # reset (set phase to 0) or random
      'selection': 'fixed',     # fixed or random
      'num2perturb': 3,         # integer for 'random' selection
      'indices': [0,1,2],       # list of integers for 'fixed' selection
      'size': 2,                # float, used only when type='random'
      'IC': 0*np.random.rand(num_osc)*np.pi*2  # initial condition for first repeat
      }


solution_params={'dt': dt, 
                 'tmax': tmax,
                 'noise': noise_level,
                 'dynamic noise': dynamic_noise_level,
                 'ts_skip': 1, # don't skip timesteps
                 'num_repeats': num_repeats,
                 #'IC': np.random.rand(num_osc)*np.pi*2, # fixed initial condition for each repeat
                 'IC': IC_setup
                 }

my_network.add_ivp_info(solution_params)

##############################################################################
## solve IVP and plot results

t = np.arange(0,tmax,dt)[:-1].reshape(-1,1)
n_ts = t.shape[0]

phases = my_network.gen_data()


figsize=(12,4)
fontsize=16
plt.figure(figsize=figsize)    
for rep in range(num_repeats):
    
    cur_t=t+rep*tmax
    cur_phases=phases[rep*n_ts:(rep+1)*n_ts]
    R,Psi=lk.get_op(cur_phases)
    plt.subplot(1,3,1)
    plt.plot(cur_t,cur_phases)
    plt.title('Phases',fontsize=fontsize)
    plt.xlabel('time',fontsize=fontsize)
    plt.ylabel('phases',fontsize=fontsize)
    plt.subplot(1,3,2)
    plt.plot(cur_t,R,'b')
    plt.title('Order parameter',fontsize=fontsize)
    plt.xlabel('time',fontsize=fontsize)
    plt.ylabel('R(t)=|Z(t)|',fontsize=fontsize)
    plt.ylim(0,1.1)
    plt.subplot(1,3,3)
    plt.plot(cur_t,Psi,'b')
    plt.title('Order parameter',fontsize=fontsize)
    plt.xlabel('time',fontsize=fontsize)
    plt.ylabel(r'$\Psi(t)=arg(Z(t))$',fontsize=fontsize)
    plt.ylim(-np.pi,np.pi)
    if rep>=1:
        for subplot in range(1,4):
            ax=plt.subplot(1,3,subplot)
            ylim=ax.get_ylim()
            ax.axvline(x=rep*tmax,ymin=ylim[0],ymax=ylim[1],color='k',linestyle='--')
plt.show()

num_attempts = 5 # number of times to attempt to learn from data for each network
method = 'euler' #'rk2','rk4','euler'