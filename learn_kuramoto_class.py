# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:22:47 2020

@author: Lauren
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy import optimize
#from functools import partial
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import warnings
from sklearn.metrics import roc_curve, auc, f1_score
from scipy import interpolate
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
            
        if dt is None:
            if t is None:
                self.dt = 0.1 # default if no time info given
            else:
                self.dt = t[1]-t[0] # assumed constant
                
        
    def gen_diffs(self):
        '''
        See: get_diff_mat.
        '''
        
        finaldiffmat = np.zeros(shape=(self.n_timestep,self.num_oscs,self.num_oscs))
        
        for index in range(self.n_timestep):
            row = self.phase[index,:]
            rowvec = np.array(row,ndmin=2)
            colvec = np.transpose(rowvec)
            diffmat = -(rowvec-colvec)
            finaldiffmat[index,:,:] = diffmat
        
        self.diffs = finaldiffmat
        
        
    def gen_training_test_data(self,frac=0.8):
        '''
        See: get_training_testing_data and get_split.

        Parameters
        ----------
        frac : TYPE, optional
            Proportion of data to use as training data. The default is 0.8.

        '''
        if not hasattr(self,'diffs'):
            self.gen_diffs()
        
        inds = np.random.permutation(self.n_timestep)
        stop = int(np.ceil(frac*self.n_timestep))
        traininds = inds[:stop]
        testinds = inds[stop:]
        
        self.diff_train = add_dim(self.diffs[traininds,:,:])
        self.phase_train = self.phase[traininds,:]
        self.vel_train = self.vel[traininds,:]
        
        self.diff_test = add_dim(self.diffs[testinds,:,:])
        self.phase_test = self.phase[testinds,:]
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
        if hasattr(self,'n_attempts'):
            best_error_val = float("inf")
            
            for att in range(self.n_attempts):
                A,omega,fout,K,error_val,FC_weights = self.learn_once(data)
                
                if error_val < best_error_val: # store the best outputs
                    self.A = A
                    self.omega = omega
                    self.fout = fout
                    self.K = K
                    self.error_val = error_val
                    best_error_val = error_val
                    self.FC_weights = FC_weights
            
        else:
            self.A,self.omega,self.fout,self.K,self.error_val,self.FC_weights = self.learn_once(data)
        
        return self.A,self.omega,self.fout,self.K,self.error_val,self.FC_weights
    
    def learn_once(self,data):
        
        # contruct model
        tf.reset_default_graph()
        if self.global_seed>0:
            tf.set_random_seed(self.global_seed) # remove this later
            
        # initialize placeholders for inputs
        X1 = tf.placeholder(dtype=tf.float32, shape=(None,self.num_osc,self.num_osc,1), name="X1")
        X2 = tf.placeholder(dtype=tf.float32, shape=(None,self.num_osc), name="X2")
        y = tf.placeholder(dtype=tf.float32, shape=(None,self.num_osc), name="y")
        
        
        ## initialize variable A (Adjacency matrix) that is symmetric with 0 entries on the diagonal.
        A_rand = tf.Variable(tf.random_normal((self.num_osc,self.num_osc),
                                              mean=0.5,
                                              stddev=1/self.num_osc),
                             name='A_rand',
                             dtype=tf.float32)
        
        A_upper = tf.matrix_band_part(A_rand, 0, -1)
        A = 0.5 * (A_upper + tf.transpose(A_upper))-tf.matrix_band_part(A_upper,0,0)
        
        ## initialize variable omega (natural frequencies) 
        omega = tf.Variable(tf.random_normal((1,self.num_osc),mean=0,stddev=1/self.num_osc,dtype=tf.float32),
                            name='omega',dtype=tf.float32) 
        
        ## initialize variable K (coupling strength value)
        K = tf.Variable(tf.random_normal(shape=(1,),mean=1,stddev=1/self.num_osc,dtype=tf.float32),name='K') 
        
        
        c = np.array([1.0,1.0]) # regularization parameters for A matrix
        
        ## compute phase velocities
        v,fout = self.get_vel(A,omega,K,X1)
        
        ## compute predictions
        if self.prediction_method=='rk2':
            k1=v
            k2=self.get_vel(A,omega,K,self.get_diff_tensor(X2+data.dt*k1/2.0))[0] # compute improved velocity prediction
            velpred=k2
        elif self.prediction_method=='rk4':
            k1=v
            k2=self.get_vel(A,omega,K,self.get_diff_tensor(X2+data.dt*k1/2.0))[0]
            k3=self.get_vel(A,omega,K,self.get_diff_tensor(X2+data.dt*k2/2.0))[0]
            k4=self.get_vel(A,omega,K,self.get_diff_tensor(X2+data.dt*k3))[0]
            velpred=1/6.0*k1+1/3.0*k2+1/3.0*k3+1/6.0*k4
        elif self.prediction_method=='euler':
            velpred=v
        else:
            print('Invalid prediction method. Using default of Euler.')
            velpred=v
        
        ## compute regularization terms for neural network weights
        l2_loss = tf.losses.get_regularization_loss()
        
        ## loss function computation
        with tf.name_scope("loss"):
            loss = self.loss_sse(velpred,y,A,c) + l2_loss
        
        ## initialize optimizer (use Adam)
        with tf.name_scope("train"):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                               beta1=0.9,beta2=0.999)
            training_op = optimizer.minimize(loss)
            
        ## compute error to be displayed (currently ignores regularization terms)
        with tf.name_scope("eval"):
            error = self.loss_sse(velpred,y,A,np.array([0.0,0.0])) # no Aij error away from 0,1
            
        
        init = tf.global_variables_initializer()
        
        ## initialize variables and optimize variables
        with tf.Session() as sess:
            init.run()
            
            ## loop for batch gradient descent
            for epoch in range(self.n_epochs):
                for X1_batch, X2_batch, y_batch in data.shuffle_batch(self.batch_size):
                    sess.run(training_op, feed_dict={X1: X1_batch, X2: X2_batch, y: y_batch})
                error_batch = error.eval(feed_dict={X1: X1_batch, X2: X2_batch, y: y_batch})
                error_val = error.eval(feed_dict={X1: data.diff_test, X2: data.phase_test, y: data.vel_test})
                
                ## display results every 20 epochs
                if epoch % 20==0:
                    print('',end='\n')
                    print("Epoch:",epoch, "Batch error:", error_batch, "Val error:", error_val,end='')
                else:
                    print('.',end='')
                    #print(tf.trainable_variables())
            print('',end='\n')
            
            self.diff_mesh = np.linspace(-np.pi,np.pi,100*self.num_osc*self.num_osc,
                                         endpoint=True).reshape(
                                             (100,self.num_osc,self.num_osc,1))
                                             
            
            FC_weights = tf.get_default_graph().get_tensor_by_name(
                "fourier0/kernel:0").eval()[0,0,:,0]
            
            return(A.eval(),
                   omega.eval(),
                   fout.eval(feed_dict={X1: self.diff_mesh, X2: data.phase_test, 
                                        y: data.vel_test}),
                   K.eval(),
                   error_val,
                   FC_weights)
    
    
    
    def loss_sse(self,ypred,ytrue,A,c=[1.0,1.0]):
        
        push_in = 100
        push_out = 10**(-6)
        loss=(tf.reduce_mean(tf.square(tf.subtract(ypred,ytrue)),name="loss")
                +push_in*(c[0]*tf.reduce_mean(tf.maximum(A-1,0))
                      +c[1]*tf.reduce_mean(tf.maximum(-A,0)))
                +push_out*(tf.reduce_mean(tf.square(A))
                           +tf.reduce_mean(tf.square(1-A))))
        
        return loss
    
    
    
    def get_vel(self,A,omega,K,X):
        
        G = self.single_network(X)
        v = omega + K*tf.reduce_mean(tf.multiply(A,G),axis=1)
        
        return v,G
    
    def single_network(self,X):
        
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg)
        initializer = tf.zeros_initializer() # from MP
        #prev_weights = np.random.normal(0, 1, (1,1,10,1)) # uses tensor shape
        #initializer = tf.keras.initializers.constant(prev_weights)
        
        Xmerged = self.fourier_terms(X)
        with tf.name_scope("fourier"):
            fout = tf.layers.conv2d(inputs=Xmerged,
                                    filters=1,
                                    kernel_size=[1, 1],
                                    padding="same",
                                    strides=(1,1),
                                    activation=None,
                                    name="fourier0",
                                    kernel_regularizer=regularizer,
                                    kernel_initializer=initializer, # changed
                                    use_bias=False,
                                    reuse=tf.AUTO_REUSE,
                                    trainable=True
                                    )
        return tf.cast(tf.squeeze(fout),tf.float32)
    
    def fourier_terms(self,X):
        
        Xmerged = tf.concat([tf.sin(X),tf.cos(X)],axis=3)
        for n in range(2,self.n_coeffs+1):
            Xmerged = tf.concat([Xmerged,tf.sin(n*X),tf.cos(n*X)],axis=3)
        return Xmerged
    
    def get_diff_tensor(self,y):
        
        finaldiffmat = tf.reshape(y,[-1,self.num_osc,1,1])-tf.reshape(y,[-1,1,self.num_osc,1])
        return finaldiffmat
    
    
    
    def evaluate(self,true_net, print_results=True, show_plots=False):
        
        if self.K < 0:
            self.fout = -1.0*self.fout
            self.K = -1.0*self.K
        
        f_res,c = evaluate_f(self.diff_mesh, self.fout, self.K, true_net.Gamma, self.FC_weights, 
                           print_results=print_results, show_plots=show_plots)
        print(c[0],'+',c[1],'*f minimizes difference in area')
        A_res = evaluate_A(self.A, true_net.A, proportion_of_max=0.9,
                         print_results=print_results, show_plots=show_plots)
        
        Nj = (self.A/c[1]).sum(axis=0)
        self.omega = self.omega - self.K*Nj*c[0]/self.num_osc
        w_res = evaluate_w(self.omega, true_net.w, print_results=print_results)
        
        return f_res, A_res, w_res
        
    
def gamma_inf_FC(weights,ins):
    '''
    interprets vector of Fourier coefficients into truncated Fourier series
    for evaluation of the inferred interaction function
    '''
    
    outs = np.zeros(ins.shape)
    n_coeffs = len(weights)/2
    
    for n in range(1,n_coeffs + 1):
        outs = outs + weights[2*n-2]*np.sin(n*ins)
        outs = outs + weights[2*n-1]*np.cos(n*ins)
    
    return outs
    

def add_dim(X,axis=3):
    '''
    add_dim(X,axis=3):
        add dimension to tensor
    '''
    return np.expand_dims(X,axis).copy()


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
    
    w_res = pd.Series(dtype='float64')
    w_res['Maximum absolute deviation'] = np.max(absolute_deviation)
    w_res['Mean absolute deviation'] = np.mean(absolute_deviation)
    w_res['Maximum relative deviation (%)'] = np.max(relative_deviation)
    w_res['Mean relative deviation (%)'] = np.mean(relative_deviation)
    w_res['Correlation'] = np.corrcoef(np.concatenate([truew,predw],axis=1).T)[0,1]
    
    return w_res

def evaluate_f(testX1,fout,K,corr_G,weights, print_results=True,show_plots=False):
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
        plt.plot(x_for_fout,c[0]+c[1]*gamma_inf_FC(weights,x_for_fout),'green')
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
    
    A_res=pd.Series(dtype='float64')
    
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

