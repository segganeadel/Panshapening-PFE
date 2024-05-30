import numpy as np

def ERGAS( I_FU:np.ndarray, I_GT:np.ndarray, ratio=4) -> float:
    """
    Parameters
    ----------
    I_FU : np.ndarray
        Fused image of shape H x W x C
        H: Height of the image, W: Width of the image, C: Number of channels
    I_GT : np.ndarray
        Ground truth image of shape H x W x C
        H: Height of the image, W: Width of the image, C: Number of channels
    ratio : int
        Ratio of spatial resolution between the low and high resolution images
    Returns
    -------
    float
        ERGAS index
    """
    return (100/ratio) * np.sqrt(np.mean(np.mean((I_GT-I_FU)**2,axis=(1,2))/(np.mean(I_GT, axis=(1,2)))**2))