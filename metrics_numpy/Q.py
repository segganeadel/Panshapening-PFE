from skimage.metrics import structural_similarity as ssim
import numpy as np

def Q(I1,I2,S):

    Q_orig = np.zeros((I1.shape[2],1))
    
    for idim in range(I1.shape[2]):
        Q_orig[idim] = ssim(I1[:,:,idim],I2[:,:,idim], win_size=S, data_range=2047.0)

    return np.mean(Q_orig)