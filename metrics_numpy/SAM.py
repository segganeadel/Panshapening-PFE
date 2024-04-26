import numpy as np

def SAM(I1:np.ndarray, I2:np.ndarray, epsilon= 2 * 10**(-16)) -> tuple[float, np.ndarray]:
    """
    Parameters
    ----------
    I1 : np.ndarray
        Ground truth image of shape H x W x C
        H: Height of the image, W: Width of the image, C: Number of channels
    I2 : np.ndarray
        Fused image of shape H x W x C
        H: Height of the image, W: Width of the image, C: Number of channels
    epsilon : float
        Small value to avoid division by zero
    Returns
    -------
    tuple[float, np.ndarray]
        SAM index and the SAM map
    """
    I1 = I1.astype('float64')
    I2 = I2.astype('float64')

    prod_scal = np.sum(I1 * I2, axis=0)
    norm_orig = np.sum(I1**2, axis=0)
    norm_fusa = np.sum(I2**2, axis=0)
    lower_term = np.sqrt(norm_orig * norm_fusa)
    lower_term = np.where(lower_term == 0, epsilon, lower_term) # to avoid division by zero
    SAM_map = np.arccos(prod_scal/lower_term)
    angolo = np.mean(SAM_map)
    SAM_index = np.rad2deg(angolo)
    return SAM_index, SAM_map