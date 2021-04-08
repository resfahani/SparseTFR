#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Sep  8 18:03:23 2020

@author: reza D. D. Esfahani


y:  1D vector [n,1]
W: window matrix [n,n]
epsilon: Nonnegative scallar of noise level (N* sigma^2)
cgIter: number of CG iteration
gamma: The weight of L2 norm
verbose: Display the residual

##################################################3
The code is part of
  "Sparsity promoting method to estimate the dispersion curve of surface wave group velocity
" paper ,DOI: 10.1190/geo2018-0138.1


and the orginal paper is 
"Sparse Time-Frequency Decomposition and Some Applications
"June 2013 IEEE Transactions on Geoscience and Remote Sensing by Ali Gholami

"""

from numpy.fft import (fft, ifft)
import numpy as np



def sparseTF(y, W, epsilon, gamma, verbose = True, cgIter = 1):

    mu = 0.5
    lam = 0.25
    
    weight = max(abs(y[:]))
    y = y[:] / weight
    n = len(y)
    B = lambda X : mu * W * np.tile(np.sum(W * X, axis = 1, keepdims = True), n) + (lam + gamma) * X
    
    
    p = np.zeros([n, n])
    q = np.zeros(p.shape)
    yk = np.zeros(y.shape)
    k = 1
    
#-------------------------------------------------------------------------
# Bregman Iterations (main loop)
#-------------------------------------------------------------------------

    while 1:
    
        rhs = mu * W * np.tile(yk, n) +  lam * ifft(p - 2 * q, axis = 0)
        u   = CG(B, rhs, 1e-4, cgIter)
        p   = fft(u, axis = 0 ) + q
        q   = clip(p, 1/lam)
        dy  = y - (np.sum(W * u , keepdims = True, axis = 1))
        yk  = yk + dy
        res = np.real(np.conj(dy.T) @ dy)
        
        if verbose:
            
            print("Iteration = " + str(k) + ", residual = " + str(np.round(res.item(), 4)) + " ("+str(epsilon) + ")" )
    
        k  = k + 1
    
        if res < epsilon:
    
            break
    
    ap = abs(p).copy()
    b  = ap - 1/lam
    f  = p * ((b + abs(b)) /ap/2)
    f  = weight * f
    
    
    return f, u


def clip(p, tau):
    p2 = np.real(p * np.conj(p))
    indx = p2 > tau**2
    q = p.copy()
    q[indx] = tau * (p[indx] / np.sqrt(p2[indx]))
    return q


def CG(A, b, tol , maxiter):
    
    
    x = np.zeros(np.shape(b))
    r = b.copy()
    d = r.copy()
    
    delta = np.sum(np.sum((np.conj(r) * r )))
    delta0 = delta.copy()
    numiter = 0
    
    
    while (numiter< maxiter) & (delta > (tol**2) * delta0):

        q = A(d)
        
        alpha = delta / np.sum(np.sum((np.conj(d) * q )))

        x = x + alpha * d
        r = r - alpha * q
        
        deltaold = delta.copy()
        
        delta = np.sum(np.sum((np.conj(r) * r)))
        
        beta = delta / deltaold
        d = r + beta * d
        
        numiter = numiter + 1

    return x
        
        
    
    
