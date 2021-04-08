#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Sep  8 18:03:23 2020

@author: reza D. D. Esfahani


The code is part of
  "Sparsity promoting method to estimate the dispersion curve of surface wave group velocity
" paper ,DOI: 10.1190/geo2018-0138.1


and the orginal paper is 
"Sparse Time-Frequency Decomposition and Some Applications
"June 2013 IEEE Transactions on Geoscience and Remote Sensing by Ali Gholami

"""


import matplotlib.pyplot as plt

import numpy as np
from scipy.linalg import toeplitz

from sparseTF import sparseTF
from numpy.fft import (fft, ifft)


# syntetic Data 
#y:  1D vector [n,1]

y = np.load("/home/reza/Seismology work/Projects/Velocity Changes/STFT/Data/" +"/signalSTFT.npy")



#gamma: The weight of L2 norm
gamma = 0.01

#cgIter: number of CG iteration
cgIter = 10

#epsilon: Nonnegative scallar of noise level (N* sigma^2)
epsilon = 1e-3


#W: window matrix [n,n]
#Window length
nw = 30
n = len(y)
t = (np.arange(0, 2*nw) - (nw)) /nw/2
w = np.exp(- np.power(t, 2)* 18 )

w = np.concatenate([w[nw:], np.zeros(n+nw*2), w[:nw]])
W = toeplitz(w)
W = W[:n,:n] / np.linalg.norm(w)
#

# Sparse Time-Frequency representation
f,u = sparseTF(y, W, epsilon,  gamma, verbose = True, cgIter = cgIter)

# Signal reconstruction from TF domain
yrec  = np.real(np.sum(W * ifft(f, axis=0) , keepdims=True, axis=1))

#%%
fig, (ax,ax1,ax2) = plt.subplots(3,1,figsize=(7,8))

t = np.arange(n)


fmin = 0
nyqn = n//2
ax.plot(t, y,'k', lw = 1)
ax.set_title('Signal', fontsize=16)
ax.set_xticks([])
ax.set_xlim(0, n)

ax1.imshow(abs(f[nyqn+1:,:]), aspect= 'auto', extent= [t[0], t[-1],fmin, nyqn], cmap='hot_r')
ax1.set_title('Sparse TF representation', fontsize=16)
ax1.set_ylabel('Frequency sample', fontsize=16)
ax1.set_xticks([])

ax2.plot(y-yrec,'k', lw = 1)
ax2.set_title('Residual', fontsize=16)
ax2.set_xlabel('Time', fontsize=16)
ax2.set_xlim(0, n)

fig.savefig("Demo1.png", dpi=100)
