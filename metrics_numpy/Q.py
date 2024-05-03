from skimage.metrics import structural_similarity as ssim
import numpy as np

def Q(gt_ms:np.ndarray, fus_ms:np.ndarray, window_size = 33, data_range = 1.0):
    """
    Parameters:
    -----------
    gt_ms: np.ndarray
        Multispectral image (ground-truth) shape (H x W x C).
    fus_ms: np.ndarray
        Multispectral image (fused) shape (H x W x C).
    window_size: int
        Size of the sliding window needs to be odd, by default 33.
    data_range: float
        The data range of the input image (distance between minimum and maximum possible values), by default 1.0 for normalized images.
    Returns:
    --------
    float
        Average Q-index value.
    """

    Q_orig = ssim(gt_ms, fus_ms, win_size=window_size, data_range=1.0, channel_axis= 2)

    return np.mean(Q_orig)