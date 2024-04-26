# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description: 
           Q2n index. 
 
 Interface:
           [Q2n_index, Q2n_index_map] = q2n(I_GT, I_F, Q_blocks_size, Q_shift)

 Inputs:
           I_GT:               Ground-Truth image;
           I_F:                Fused Image;
           Q_blocks_size:      Block size of the Q-index locally applied;
           Q_shift:            Block shift of the Q-index locally applied.

 Outputs:
           Q2n_index:          Q2n index;
           Q2n_index_map:      Map of Q2n values.

 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.
"""

import math
import numpy as np

def q2n(I_GT, I_F, Q_blocks_size, Q_shift):

    N1 = I_GT.shape[0]
    N2 = I_GT.shape[1]
    N3 = I_GT.shape[2]
    
    size2 = Q_blocks_size
    
    stepx = math.ceil(N1/Q_shift)
    stepy = math.ceil(N2/Q_shift)
     
    if (stepy <= 0):
        stepy = 1
        stepx = 1
    
    est1 = (stepx - 1) * Q_shift + Q_blocks_size - N1
    est2 = (stepy - 1) * Q_shift + Q_blocks_size - N2
       
    if (est1 != 0) or (est2 != 0):
      refref = []
      fusfus = []
      
      for i in range(N3):
          a1 = np.squeeze(I_GT[:,:,0])
        
          ia1 = np.zeros((N1+est1,N2+est2))
          ia1[0:N1,0:N2] = a1
          ia1[:,N2:N2+est2] = ia1[:,N2-1:N2-est2-1:-1]
          ia1[N1:N1+est1,:] = ia1[N1-1:N1-est1-1:-1,:]

          if i == 0:
              refref = ia1
          elif i == 1:
              refref = np.concatenate((refref[:,:,None],ia1[:,:,None]),axis=2)
          else:
              refref = np.concatenate((refref,ia1[:,:,None]),axis=2)
          
          if (i < (N3-1)):
              I_GT = I_GT[:,:,1:I_GT.shape[2]]
          
      I_GT = refref
            
      for i in range(N3):
          a2 = np.squeeze(I_F[:,:,0])
          
          ia2 = np.zeros((N1+est1,N2+est2))
          ia2[0:N1,0:N2] = a2
          ia2[:,N2:N2+est2] = ia2[:,N2-1:N2-est2-1:-1]
          ia2[N1:N1+est1,:] = ia2[N1-1:N1-est1-1:-1,:]
          
          if i == 0:
              fusfus = ia2
          elif i == 1:
              fusfus = np.concatenate((fusfus[:,:,None],ia2[:,:,None]),axis=2)
          else:
              fusfus = np.concatenate((fusfus,ia2[:,:,None]),axis=2)
          
          if (i < (N3-1)):
              I_F = I_F[:,:,1:I_F.shape[2]]
          
      I_F = fusfus
      
    #I_F = np.uint16(I_F)
    #I_GT = np.uint16(I_GT)
    
    N1 = I_GT.shape[0]
    N2 = I_GT.shape[1]
    N3 = I_GT.shape[2]
    
    if (((math.ceil(math.log2(N3))) - math.log2(N3)) != 0):
        Ndif = (2**(math.ceil(math.log2(N3)))) - N3
        dif = np.zeros((N1,N2,Ndif))
        #dif = np.uint16(dif)
        I_GT = np.concatenate((I_GT, dif), axis = 2)
        I_F = np.concatenate((I_F, dif), axis = 2)
    
    N3 = I_GT.shape[2]
    
    valori = np.zeros((stepx,stepy,N3))
    
    for j in range(stepx):
        for i in range(stepy):
            o = onions_quality(I_GT[ (j*Q_shift):(j*Q_shift)+Q_blocks_size, (i*Q_shift):(i*Q_shift)+size2,:], I_F[ (j*Q_shift):(j*Q_shift)+Q_blocks_size,(i*Q_shift):(i*Q_shift)+size2,:], Q_blocks_size)
            valori[j,i,:] = o    
        
    Q2n_index_map = np.sqrt(np.sum(valori**2, axis=2))

    Q2n_index = np.mean(Q2n_index_map)
    
    return Q2n_index, Q2n_index_map


import numpy as np


def onions_quality(dat1,dat2,size1):

    #dat1 = dat1.astype('float64')
    #dat2 = dat2.astype('float64')
    
    
    h = dat2[:,:,0]
    dat2 = np.concatenate((h[:,:,None],-dat2[:,:,1:dat2.shape[2]]),axis=2)
    
    N3 = dat1.shape[2]
    size2 = size1
    
    """ Block normalization """
    for i in range(N3):
      a1,s,t = norm_blocco(np.squeeze(dat1[:,:,i]))
      dat1[:,:,i] = a1
      
      if (s == 0):
          if (i == 0):
              dat2[:,:,i] = dat2[:,:,i] - s + 1
          else:
              dat2[:,:,i] = -(-dat2[:,:,i] - s + 1)
      else:
          if (i == 0):
              dat2[:,:,i] = (dat2[:,:,i] - s)/t + 1
          else:
              dat2[:,:,i] = -(((-dat2[:,:,i] - s)/t) + 1)    
    
    m1 = np.zeros((1,N3))
    m2 = np.zeros((1,N3))
    
    mod_q1m = 0
    mod_q2m = 0
    mod_q1 = np.zeros((size1,size2))
    mod_q2 = np.zeros((size1,size2))
    
    for i in range(N3):
        m1[0,i] = np.mean(np.squeeze(dat1[:,:,i]))
        m2[0,i] = np.mean(np.squeeze(dat2[:,:,i]))
        mod_q1m = mod_q1m + m1[0,i]**2
        mod_q2m = mod_q2m + m2[0,i]**2
        mod_q1 = mod_q1 + (np.squeeze(dat1[:,:,i]))**2
        mod_q2 = mod_q2 + (np.squeeze(dat2[:,:,i]))**2
    
    mod_q1m = np.sqrt(mod_q1m)
    mod_q2m = np.sqrt(mod_q2m)
    mod_q1 = np.sqrt(mod_q1)
    mod_q2 = np.sqrt(mod_q2)
    
    termine2 = mod_q1m * mod_q2m
    termine4 = mod_q1m**2 + mod_q2m**2
    int1 = (size1 * size2)/((size1 * size2)-1) * np.mean(mod_q1**2)
    int2 = (size1 * size2)/((size1 * size2)-1) * np.mean(mod_q2**2)
    termine3 = int1 + int2 - (size1 * size2)/((size1 * size2) - 1) * ((mod_q1m**2) + (mod_q2m**2))
    
    mean_bias = 2*termine2/termine4
    
    if (termine3==0):
        q = np.zeros((1,1,N3))
        q[:,:,N3-1] = mean_bias
    else:
        cbm = 2/termine3
        qu = onion_mult2D(dat1, dat2)
        qm = onion_mult(m1, m2)
        qv = np.zeros((1,N3))
        for i in range(N3):
            qv[0,i] = (size1 * size2)/((size1 * size2)-1) * np.mean(np.squeeze(qu[:,:,i]))
        q = qv - ((size1 * size2)/((size1 * size2) - 1.0)) * qm
        q = q * mean_bias * cbm
    return q

def onion_mult2D(onion1,onion2):

    N3 = onion1.shape[2]

    if (N3 > 1):
        L = int(N3/2)
        a = onion1[:,:,0:L]
        b = onion1[:,:,L:onion1.shape[2]]
        h = b[:,:,0]
        b = np.concatenate((h[:,:,None],-b[:,:,1:b.shape[2]]),axis=2)
        c = onion2[:,:,0:L]
        d = onion2[:,:,L:onion2.shape[2]]
        h = d[:,:,0]
        d = np.concatenate((h[:,:,None],-d[:,:,1:d.shape[2]]),axis=2)

        if (N3 == 2):
            ris = np.concatenate((a*c-d*b,a*d+c*b),axis=2)
        else:
            ris1 = onion_mult2D(a,c)
            h = b[:,:,0]
            ris2 = onion_mult2D(d,np.concatenate((h[:,:,None],-b[:,:,1:b.shape[2]]),axis=2))
            h = a[:,:,0]
            ris3 = onion_mult2D(np.concatenate((h[:,:,None],-a[:,:,1:a.shape[2]]),axis=2),d)
            ris4 = onion_mult2D(c,b)
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.concatenate((aux1,aux2), axis=2)
    else:
        ris = onion1 * onion2   

    return ris

def onion_mult(onion1,onion2):

    N = onion1.size
    
    if (N > 1):      
        L = int(N/2)   
        
        a = onion1[0,0:L]
        b = onion1[0,L:onion1.shape[1]]
        b = np.concatenate(([b[0]],-b[1:b.shape[0]]))
        c = onion2[0,0:L]
        d = onion2[0,L:onion2.shape[1]]
        d = np.concatenate(([d[0]],-d[1:d.shape[0]]))
    
        if (N == 2):
            ris = np.concatenate((a*c-d*b, a*d+c*b))
        else:
            ris1 = onion_mult(np.reshape(a,(1,a.shape[0])),np.reshape(c,(1,c.shape[0])))
            ris2 = onion_mult(np.reshape(d,(1,d.shape[0])),np.reshape(np.concatenate(([b[0]],-b[1:b.shape[0]])),(1,b.shape[0])))
            ris3 = onion_mult(np.reshape(np.concatenate(([a[0]],-a[1:a.shape[0]])),(1,a.shape[0])),np.reshape(d,(1,d.shape[0])))
            ris4 = onion_mult(np.reshape(c,(1,c.shape[0])),np.reshape(b,(1,b.shape[0])))
    
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
    
            ris = np.concatenate((aux1,aux2))
    else:
        ris = onion1 * onion2

    return ris

def norm_blocco(x):
    
    a = np.mean(x)
    c = np.std(x, ddof=1)
    
    if (c==0):
        c = 2.2204 * 10**(-16)
    
    y = ((x - a)/c) + 1
    
    return y, a, c