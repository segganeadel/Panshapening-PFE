import numpy as np

def SAM(I1, I2):

    I1 = I1.astype('float64')
    I2 = I2.astype('float64')

    prod_scal = np.sum(I1 * I2, axis=0)
    norm_orig = np.sum(I1**2, axis=0)
    norm_fusa = np.sum(I2**2, axis=0)
    lower_term = np.sqrt(norm_orig * norm_fusa)
    lower_term = np.where(lower_term == 0, 2 * 10**(-16), lower_term) # to avoid division by zero
    SAM_map = np.arccos(prod_scal/lower_term)
    angolo = np.mean(SAM_map)
    SAM_index = np.rad2deg(angolo)
    return SAM_index, SAM_map