import numpy as np

def ERGAS(I_GT, I_FU, ratio=4):
    """
    Input shape C x H x W
    """
    I_GT = I_GT.astype('float64')
    I_FU = I_FU.astype('float64')
     
    return (100/ratio) * np.sqrt(np.mean(np.mean((I_GT-I_FU)**2,axis=(1,2))/(np.mean(I_GT, axis=(1,2)))**2))