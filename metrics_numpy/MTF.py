from scipy.ndimage.filters import correlate, gaussian_filter

import numpy as np

def MTF(I_MS, sensor, ratio):
    
    h = genMTF(ratio, sensor,I_MS.shape[2])
    
    I_MS_LP = np.zeros((I_MS.shape))
    for ii in range(I_MS.shape[2]):
        I_MS_LP[:,:,ii] = correlate(I_MS[:,:,ii],h[:,:,ii],mode='nearest')
        ### This can speed-up the processing, but with slightly different results with respect to the MATLAB toolbox
        # hb = h[:,:,ii]
        # I_MS_LP[:,:,ii] = signal.fftconvolve(I_MS[:,:,ii],hb[::-1],mode='same')

    return np.double(I_MS_LP)

def genMTF(ratio, sensor, nbands):
    
    GNyq_dict_ms = {
        'QB':       [0.34, 0.32, 0.30, 0.22], # Band Order: B,G,R,NIR
        'IKONOS':   [0.26, 0.28, 0.29, 0.28],    # Band Order: B,G,R,NIR
        'GeoEye1':  [0.23, 0.23, 0.23, 0.23],    # Band Order: B,G,R,NIR
        'WV2':      [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.27],
        'WV3':      [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]
    } 
    GNyq = GNyq_dict_ms.get(sensor, 0.3 * np.ones(nbands)) # Default value
    GNyq = np.asarray(GNyq, dtype=np.float32) # Ensure that GNyq is a numpy array
    
    kernel_size = 41  
  
    """MTF"""
    fcut = 1/ratio
    h = np.zeros((kernel_size,kernel_size,nbands))
    alphas = np.sqrt(((kernel_size-1)*(fcut/2))**2/(-2*np.log(GNyq)))
    
    for i in range(nbands):
        H   = gaussian2d(kernel_size, alphas[i])
        Hd  = H / np.max(H)
        w   = kaiser2d(kernel_size, 0.5)
        h[:,:,i] = np.real(fir_filter_wind(Hd,w))
        
    return h

def fir_filter_wind(Hd,w):
    
    hd=np.rot90(np.fft.fftshift(np.rot90(Hd,2)),2)
    h=np.fft.fftshift(np.fft.ifft2(hd))
    h=np.rot90(h,2)
    h=h*w
    #h=h/np.sum(h)
    
    return h

def gaussian2d (size, sigma):
    

    t=np.arange(-(size-1)/2, (size+1)/2)
    t1,t2=np.meshgrid(t,t)

    sigma=np.double(sigma)
    w = np.exp(-0.5*(t1/sigma)**2)*np.exp(-0.5*(t2/sigma)**2) 

    return w
   
def kaiser2d (size, beta):

    t=np.arange(-(size-1)/2,(size+1)/2) /np.double(size-1)
    t1,t2=np.meshgrid(t,t)
    t12=np.sqrt(t1*t1+t2*t2)

    w1=np.kaiser(size,beta)
    w=np.interp(t12,t,w1)
    w[t12>t[-1]]=0
    w[t12<t[0]]=0
    
    return w
