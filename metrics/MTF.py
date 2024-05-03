# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description: 
           MTF filters the image I_MS using a Gaussin filter matched with the Modulation Transfer Function (MTF) of the MultiSpectral (MS) sensor. 
 
 Interface:
           I_Filtered = MTF(I_MS,sensor,ratio)

 Inputs:
           I_MS:           MS image;
           sensor:         String for type of sensor (e.g. 'WV2', 'IKONOS');
           ratio:          Scale ratio between MS and PAN.

 Outputs:
           I_Filtered:     Output filtered MS image.
 
 Notes:
     The bottleneck of this function is the function scipy.filters.correlate that gets the same results as in the MATLAB toolbox
     but it is very slow with respect to fftconvolve that instead gets slightly different results

 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.           
"""

import scipy
import numpy as np

def MTF(I_MS,sensor,ratio):
    
    h = genMTF(ratio, sensor,I_MS.shape[2])
    
    I_MS_LP = np.zeros((I_MS.shape))
    for ii in range(I_MS.shape[2]):
        I_MS_LP[:,:,ii] = scipy.ndimage.filters.correlate(I_MS[:,:,ii],h[:,:,ii],mode='nearest')
        ### This can speed-up the processing, but with slightly different results with respect to the MATLAB toolbox
        # hb = h[:,:,ii]
        # I_MS_LP[:,:,ii] = signal.fftconvolve(I_MS[:,:,ii],hb[::-1],mode='same')

    return np.double(I_MS_LP)

def genMTF(ratio, sensor, nbands):
    
    N = 41
        
    if sensor=='QB':
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='IKONOS':
        GNyq = np.asarray([0.26,0.28,0.29,0.28],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='GeoEye1' or sensor=='WV4':
        GNyq = np.asarray([0.23,0.23,0.23,0.23],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='WV2':
        GNyq = [0.35*np.ones(nbands),0.27]
    elif sensor=='WV3':
        GNyq = np.asarray([0.325,0.355,0.360,0.350,0.365,0.360,0.335,0.315],dtype='float32') 
    else:
        GNyq = 0.3 * np.ones(nbands)
        
    """MTF"""
    h = np.zeros((N, N, nbands))

    fcut = 1/ratio

    h = np.zeros((N,N,nbands))
    for ii in range(nbands):
        alpha = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyq[ii])))
        H=gaussian2d(N,alpha)
        Hd=H/np.max(H)
        w=kaiser2d(N,0.5)
        h[:,:,ii] = np.real(fir_filter_wind(Hd,w))
        
    return h

import numpy as np

def fir_filter_wind(Hd,w):
    
    hd=np.rot90(np.fft.fftshift(np.rot90(Hd,2)),2)
    h=np.fft.fftshift(np.fft.ifft2(hd))
    h=np.rot90(h,2)
    h=h*w
    #h=h/np.sum(h)
    
    return h

def gaussian2d (N, std):
    t=np.arange(-(N-1)/2,(N+1)/2)
    #t=np.arange(-(N-1)/2,(N+2)/2)
    t1,t2=np.meshgrid(t,t)
    std=np.double(std)
    w = np.exp(-0.5*(t1/std)**2)*np.exp(-0.5*(t2/std)**2) 
    return w
    
def kaiser2d (N, beta):
    t=np.arange(-(N-1)/2,(N+1)/2)/np.double(N-1)
    #t=np.arange(-(N-1)/2,(N+2)/2)/np.double(N-1)
    t1,t2=np.meshgrid(t,t)
    t12=np.sqrt(t1*t1+t2*t2)
    w1=np.kaiser(N,beta)
    w=np.interp(t12,t,w1)
    w[t12>t[-1]]=0
    w[t12<t[0]]=0
    
    return w
