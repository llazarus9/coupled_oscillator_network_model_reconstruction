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
        
        self.phases = phases_all
        self.derivs = deriv_all
        
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

        '''
        if self.diffs is None:
            self.gen_diffs
        
        n_timestep = self.diffs.shape[0]
        inds = np.random.permutation(n_timestep)
        stop = int(np.ceil(frac*n_timestep))
        traininds = inds[:stop]
        testinds = inds[stop:]
        
        self.phase_train = self.phase[traininds,:]
        self.diff_train = self.diffs[traininds,:,:]
        self.vel_train = self.vel[traininds,:]
        
        self.phase_test = self.phase[testinds,:]
        self.diff_test = self.diffs[testinds,:,:]
        self.vel_test = self.vel[testinds,:]
        
        #return self.phase_train,self.diff_train,self.vel_train,
            #self.phase_test,self.diff_test,self.vel_test
    
    
    def shuffle_batch(self,batch_size):
        
        rnd_idx = np.random.permutation(len(self.phase_train))
        n_batches = len(self.phase_train) // batch_size
        
        for batch_idx in np.array_split(rnd_idx, n_batches):
            
            diff_batch = self.diff_train[batch_idx]
            phase_batch = self.phase_train[batch_idx]
            vel_batch = self.vel_train[batch_idx]
            
            yield diff_batch, phase_batch, vel_batch


class LearnModel():
    def __init__(self,learning_params):
        
        for key, value in learning_params.items():
            setattr(self, key, value)
    
    
    def learn(self,data):
        
        # obviously a lot to fill in here.
        
        self.diff_test = data.diff_test
        pass
    
    
    def evaluate(self,true_net, print_results=True, show_plots=False):
        
        if self.K < 0:
            self.fout = -1.0*self.fout
            self.K = -1.0*self.K
        
        f_res,c = evaluate_f(self.diff_test, self.fout, self.K, true_net.Gamma, 
                           print_results=print_results, show_plots=show_plots)
        A_res = evaluate_A(self.A, true_net.A, proportion_of_max=0.9,
                         print_results=print_results, show_plots=show_plots)
        
        Nj = (self.A/c[1]).sum(axis=0)
        self.w = self.w - self.K*Nj*c[0]/self.num_osc
        w_res = evaluate_w(self.w, true_net.w, print_results=print_results)
        
        return f_res, A_res, w_res
        
    
def loss_sse(self,ypred,ytrue,A,c=[1.0,1.0]):
    
    push_in = 100
    push_out = 10**(-6)
    loss=(tf.reduce_mean(tf.square(tf.subtract(ypred,ytrue)),name="loss")
            +push_in*(c[0]*tf.reduce_mean(tf.maximum(A-1,0))
                  +c[1]*tf.reduce_mean(tf.maximum(-A,0)))
            +push_out*(tf.reduce_mean(tf.square(A))
                       +tf.reduce_mean(tf.square(1-A))))
    
    return loss


def evaluate_w(predw,truew, print_results=True):
    ''' 
    evaluate_w(predw,system_params, print_results=True):
        compute results for frequency estimation
    Inputs:
    predw: vector of estimated frequencies
    system_params: dictionary with: 
                'w': scalar or (n,1)
                'A': (n,n)
                'K': scalar 
                'Gamma': vectorized function
    print_results: boolean to determine if results should be displayed
    
    Outputs:
    w_res: series with labeled results
    
    '''
    predw = predw.reshape((-1,1))
    
    absolute_deviation = np.abs(truew-predw)
    relative_deviation = absolute_deviation/np.abs(truew)*100
    
    if print_results:
        print('')
        print('Evaluating natural frequencies:')
        print('')    
        print('Maximum absolute deviation: %.5f' % (np.max(absolute_deviation)))
        print('Mean absolute deviation: %.5f' % (np.mean(absolute_deviation)))
        print('Maximum relative deviation (%%): %.5f' % (np.max(relative_deviation)))
        print('Mean relative deviation (%%): %.5f' % (np.mean(relative_deviation)))
        print('Correlation: %.5f' % (np.corrcoef(np.concatenate([truew,predw],axis=1).T)[0,1]))
        print('')    
    
    w_res = pd.Series()
    w_res['Maximum absolute deviation'] = np.max(absolute_deviation)
    w_res['Mean absolute deviation'] = np.mean(absolute_deviation)
    w_res['Maximum relative deviation (%)'] = np.max(relative_deviation)
    w_res['Mean relative deviation (%)'] = np.mean(relative_deviation)
    w_res['Correlation'] = np.corrcoef(np.concatenate([truew,predw],axis=1).T)[0,1]
    
    return w_res

def evaluate_f(testX1,fout,K,corr_G, print_results=True,show_plots=False):
    ''' 
    evaluate_f(predw,system_params, print_results,show_plots):
        compute results for frequency estimation
    Inputs:
    testX1: matrix of phase differences
    fout: matrix of estimated coupling function values
    K: estimated coupling strength
    system_params: dictionary with: 
                'w': scalar or (n,1)
                'A': (n,n)
                'K': scalar - ignored
                'Gamma': vectorized function
    print_results: boolean to determine if results should be displayed
    show_plots: boolean to determine if result should be plotted
    
    Outputs:
    f_res: series with labeled results
    
    '''

    FS=16  # fontsize
    n_pts=1000 # points for interpolation
    
    # reshape and sort vectors
    fout_v2=np.reshape(fout,(-1,))
    X1_v2=np.angle(np.exp(1j*np.reshape(testX1,(-1,))))
    X1_v3, fout_v3=(np.array(t) for t in zip(*sorted(zip(X1_v2,fout_v2))))
    
    
    # interpolate 
    x_for_fout=np.linspace(-np.pi,np.pi,n_pts,endpoint=True)
    predF=np.interp(x_for_fout,X1_v3,fout_v3)
    correctF = corr_G(x_for_fout)
    
    # find best scaling for coupling function
    #area_diff_func=lambda c: np.trapz(np.abs(c*predF-correctF),x_for_fout)
    #res=optimize.minimize_scalar(area_diff_func,bounds=(-100,100))
    area_diff_func=lambda c: np.trapz(np.abs(c[0]+c[1]*predF-correctF),x_for_fout)
    res=optimize.minimize(area_diff_func,x0=np.array([0,1]),bounds=[(-10,10),(-100,100)])
    c=res.x    
    # compute areas 
    
    area_between_predF_correctF=area_diff_func(c)
    area_between_null_correctF=np.trapz(np.abs(correctF),x_for_fout)
    area_ratio=area_between_predF_correctF/area_between_null_correctF
    
    
    
    f_res=pd.Series()
    f_res['Area between predicted and true coupling function']=area_between_predF_correctF
    f_res['Area between true coupling function and axis']=area_between_null_correctF
    f_res['Area ratio']=area_ratio
    
    # display results
    if show_plots:
        plt.figure()
        plt.plot(x_for_fout,c[0]+c[1]*predF,'blue')
        plt.plot(x_for_fout,correctF,'red')
        plt.xlabel(r'Phase difference $\Delta\theta$',fontsize=FS)
        plt.ylabel(r'Coupling: $\Gamma(\Delta\theta)$',fontsize=FS)
    if print_results:
        print('')
        print('Evaluating coupling function:')
        print('')
        print("Area between predicted and true coupling function: %.5f" % (area_between_predF_correctF))
        print("Area between true coupling function and axis: %.5f" % (area_between_null_correctF))
        print("Area ratio: %.5f" % (area_ratio))
        print('')
    return f_res,c

def evaluate_A(predA,trueA, print_results=True,show_plots=False, proportion_of_max=0.9):
    ''' 
    evaluate_A(predA,system_params, print_results,show_plots):
        compute results for adjacency matrix estimation
    Inputs:
    predA: predicted adjacency matrix (no threshold)

    system_params: dictionary with: 
                'w': scalar or (n,1)
                'A': (n,n)
                'K': scalar 
                'Gamma': vectorized function
    print_results: boolean to determine if results should be displayed
    show_plots: boolean to determine if result should be plotted
    
    Outputs:
    A_res: series with labeled results
    
    '''
    #print("predA:",predA,type(predA))
    FS=16 # fontsize
    pos_label=1.0 # determines which label is considered a positive.
    fpr, tpr, thresholds = roc_curve(remove_diagonal(trueA,1),
                                         remove_diagonal(predA,1),
                                         pos_label=pos_label,
                                         drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    #print("roc_auc:",roc_auc,type(roc_auc))
    warnings.filterwarnings('ignore')
    f1_scores=np.array([f1_score(remove_diagonal(trueA,1),1*(remove_diagonal(predA,1)>thr)) for thr in thresholds])
    warnings.filterwarnings('default')
    optimal_f1=np.max(f1_scores)
    optimal_threshold=thresholds[np.argmax(f1_scores)]
    inds=list(np.where(f1_scores>= proportion_of_max*optimal_f1)[0])
    threshold_range=[np.min(thresholds[inds]),np.max(thresholds[inds])]
    
    if show_plots:
        plt.figure()
        plt.plot(thresholds,f1_scores,color='black')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel('threshold',fontsize=FS)
        plt.ylabel('F1 score',fontsize=FS)
        plt.fill(np.append(thresholds[inds],[threshold_range[0],threshold_range[1]]),
                 np.append(f1_scores[inds],[0.0,0.0]),color='red',alpha=0.2)
        plt.text(0.5,0.5,'>%.1f %% of peak f1 score' %(100*proportion_of_max),fontsize=FS,ha='center')
    
    n_errors=np.sum(np.sum(abs((predA>optimal_threshold).astype(int)-trueA)))/2    
    num_osc=trueA.shape[0]
    
    if print_results:
        print('')
        print('Evaluating adjacency matrix:')   
        print('')
        print('Errors: %d out of %d' % (n_errors,(num_osc*(num_osc-1)/2)))
        print('Error rate: %.5f%%' % (n_errors/(num_osc*(num_osc-1.0)/2.0)*100.0))
        print('Area under ROC curve: %.5f' % (roc_auc))
        print('Best f1 score: %.5f' %(optimal_f1))
        print('Threshold for best f1 score: %.5f' %(optimal_threshold))
        print('Threshold range for >%.1f%% of best f1 score: [%.5f,%.5f]' % (100*proportion_of_max,threshold_range[0],threshold_range[1]))
        print('')
    
    A_res=pd.Series()
    
    A_res['Number of errors']=n_errors
    A_res['Error rate']=n_errors/(num_osc*(num_osc-1)/2)*100
    A_res['Area under ROC curve']=roc_auc
    A_res['Best f1 score']=optimal_f1
    A_res['Threshold for best f1 score']=optimal_threshold
    A_res['Threshold range for >%.1f%% of best f1 score'% (100*proportion_of_max)]=threshold_range
    

    return A_res


def remove_diagonal(A,remtype=0):
    ''' 
    remove_diagonal(A,remtype):
        turn matrix into vector without the diagonal
    Inputs:
    A: square matrix
    remtype:
        0: remove diagonal only
        1: remove diagonal and subdiagonal
        -1: remove diagonal and superdiagonal
    
    Outputs:
    entrylist: vector 
    
    '''
    nr,nc=A.shape
    entrylist=[]
    for k in range(1,nr):
        if remtype>=0: # 1 for super only
            sup=list(np.diagonal(A, offset=k, axis2=1))
            entrylist=entrylist+sup
        if remtype<=0: #-1 for sub only
            sub=list(np.diagonal(A, offset=-k, axis2=1))
            entrylist=entrylist+sub
    return entrylist

