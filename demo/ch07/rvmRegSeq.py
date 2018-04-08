# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 21:31:11 2018

@author: SCUTCYF

ToDo: beta is not updated.
Sparse Bayesian Regression (RVM) using sequential algorithm
Input:
  X: d x n data
  t: 1 x n response
Output:
  model: trained model structure
  llh: loglikelihood 
reference:
Tipping and Faul. Fast marginal likelihood maximisation for sparse Bayesian models. AISTATS 2003.

"""
import numpy as np

class model(object):
    __slots__ = ["index","w0","w","alpha","beta","xbar"]
    def __init__(self,index,w0,w,alpha,beta,xbar,U):
        self.index = index
        self.w0 = w0
        self.w = w
        self.alpha = alpha
        self.beta = beta
        self.xbar = xbar

def rvmRegSeq(X, t):
    maxiter = 1000
    inf = 1000000
    llh = -inf*np.ones(maxiter)
    tol = 1e-4
    
    (d,n)=X.shape
    xbar = np.mean(X,1).reshape((X.shape[0],1))
    tbar = np.mean(t,1).reshape((t.shape[0],1))
    X = X-xbar
    t = t-tbar
    
    beta = 1/np.mean(t*t)
    alpha = inf*np.ones((d,))
    S = beta*np.multiply(X,X).sum(axis=1)
    Q = beta*(X.dot(t.transpose()))
    Sigma = np.empty((0,0))
    mu = np.empty((0,1))
    index = np.empty((0,1))
    Phi = np.empty((0,n))
    iAct = np.empty((d,3))
    for iter in range(1,maxiter):
        s = S
        q = Q
        if index.size :
            s[index] = alpha[index]*S[index]/(alpha[index]-S[index])
            q[index] = alpha[index]*Q[index]/(alpha[index]-S[index])
        theta = q*q-s
        iNew = theta.reshape(theta.size,)>0
        
        iUse = np.zeros((d,))>0
        if index.size :
            iUse[index] = 1
        
        iUpd = (iNew & iUse) # update
        iAdd = (iNew != iUpd) # add
        iDel = (iUse != iUpd) # del
        
        dllh = -inf*np.ones((d,1)) #  delta likelihood (likelihood improvement of each step, eventually approches 0.)
        if np.any(iUpd):
            alpha_ = (s[iUpd]*s[iUpd])/theta[iUpd]
            delta = 1/alpha_-1/alpha[iUpd]
            dllh[iUpd] = Q[iUpd]*Q[iUpd]*delta/(S[iUpd]*delta+1)-np.log1p(S[iUpd]*delta)
            
        if np.any(iAdd):
            dllh[iAdd] = (Q[iAdd]*Q[iAdd]-S[iAdd])/S[iAdd]+np.log(S[iAdd]/Q[iAdd]*Q[iAdd])
            
        if np.any(iDel):
            dllh[iDel] = (Q[iDel]*Q[iDel])/S[iDel]-alpha[iDel]-np.log1p(-S[iDel]/alpha[iDel])
            
        llh[iter] = np.max(dllh)
        j = np.argmax(dllh)
        if llh[iter] < tol:
            break
        
        iAct[:,0] = iUpd
        iAct[:,1] = iAdd
        iAct[:,2] = iDel
        
        # update parameters
        find = np.nonzero(iAct[j,:])[0]
        if find == 0: # update
            idx = np.nonzero(index==j)
            alpha_ = (s[j]*s[j])/theta[j]
            
            Sigma_j = Sigma[:,idx]        
            Sigma_jj = Sigma[idx,idx]
            mu_j = mu[idx]
            
            kappa = 1/(Sigma_jj+1/(alpha_-alpha[j]))
            Sigma = Sigma-kappa*(Sigma_j.dot(Sigma_j.transpose()))
            mu = mu-kappa*mu_j*Sigma_j
            
            v = beta*X.dot(Phi.transpose().dot(Sigma_j))
            S = S+kappa*v*v
            Q = Q+kappa*mu_j*v
            alpha[j] = alpha_
        elif(find==1): # add
            alpha_ = (s[j]*s[j])/theta[j]        
            Sigma_jj = 1/(alpha_+S[j])
            mu_j = Sigma_jj*Q[j]
            phi_j = X[j,:]
            
            v = beta*X.dot(Phi.transpose().dot(phi_j))
            off = -Sigma_jj*v
            Sigma = np.append(Sigma,off,axis=1) # 
            Sigma = np.append(Sigma,[off.transpose(),Sigma_jj],axis=0)
            mu = mu-mu_j*v
            mu = np.append(mu,mu_j,axis=0)
            
            e_j = phi_j-v.transpose().dot(Phi)
            v = beta*X.dot(e_j.transpose())
            S = S-Sigma_jj.dot(v*v)
            Q = Q-mu_j.dot(v)
            
            index = np.append(index,j,axis=0)
            alpha[j] = alpha_
            
        elif(find==2): # del
            idx = np.nonzero(index==j)
            Sigma_j = Sigma[:,idx]
            Sigma_jj = Sigma_j[idx,:]
            mu_j = mu[idx]
            
            Sigma = Sigma-(Sigma.dot(Sigma.transpose()))/Sigma_jj
            mu = mu-mu_j*Sigma_j/Sigma_jj
            
            v = beta*X.dot(Phi.transpose().dot(phi_j))
            S = S+v*v/Sigma_jj
            Q = Q + mu_j*v/Sigma_jj
            
            mu[idx] = None
            Sigma[:,idx] = None
            Sigma[idx,:] = None
            index[idx] = None
            alpha[j] = inf
            
        Phi = X[index,:]
            
    llh = np.cumsum(llh[1:iter+1])
    model.index = index
    model.w0 = tbar-np.multiply(mu,xbar[index]).sum()
    model.w = mu
    model.alpha = alpha[index]
    model.beta = beta
    model.xbar = xbar

    return model,llh
