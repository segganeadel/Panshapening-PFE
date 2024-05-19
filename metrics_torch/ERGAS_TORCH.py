import torch

def ergas_torch(ms_fus:torch.Tensor, ms_gt:torch.Tensor, ratio=0.25) -> torch.Tensor:
    """
    Parameters
    ----------
    ms_fus : Tensor
        Batch of the fused multispectral images of shape N x C x H x W
        N: Number of images, C: Number of channels, H: Height of the images, W: Width of the images
    ms_gt : Tensor 
        Batch of the ground truth multispectal images of shape N x C x H x W
        N: Number of images, C: Number of channels, H: Height of the images, W: Width of the images
    ratio : int, optional, default: 0.25 (1/4)
        The ratio of spatial resolution between the MS and PAN images
    Returns
    -------
    Tensor
        Array of the ERGAS indexes of the N images
    """
    return 100 * ratio * torch.sqrt(torch.mean(torch.mean((ms_gt-ms_fus)**2,axis=(2,3))/(torch.mean(ms_gt, axis=(2,3)))**2))