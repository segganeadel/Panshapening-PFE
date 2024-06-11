import numpy as np

def linstretch(image, tol = [[0.01, 0.99], [0.01, 0.99], [0.01, 0.99]]):
    """
    Linear stretching of an image.

    Parameters:
    image: numpy array
        Input image to stretch.
    tol: list of lists
        Tolerance values for each channel in the format [[low_r, high_r], [low_g, high_g], [low_b, high_b]].

    Returns:
    numpy array
        Stretched image.
    """
    image = image.astype(np.float32)
    N, M, C = image.shape
    NM = N * M
    for i in range(C):
        b = image[:, :, i].flatten()
        hist, bin_edges = np.histogram(b, bins=np.arange(np.min(b), np.max(b) + 2))

        cdf = hist.cumsum()

        t_low = bin_edges[np.searchsorted(cdf, NM * tol[i][0])]
        t_high = bin_edges[np.searchsorted(cdf, NM * tol[i][1], side='right') - 1]

        b = np.clip(b, t_low, t_high)

        b = (b - t_low) / (t_high - t_low)

        image[:, :, i] = b.reshape(N, M)

    return image
