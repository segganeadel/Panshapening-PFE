import torch

def sam_torch(ms_gt:torch.Tensor, ms_fus:torch.Tensor, epsilon= 2 * 10**(-16)) -> torch.Tensor:
    """
    Parameters
    ----------
    ms_gt : Tensor 
        Batch of the ground truth multispectal images of shape N x C x H x W
        N: Number of images, C: Number of channels, H: Height of the images, W: Width of the images
    ms_fus : Tensor
        Batch of the fused multispectral images of shape N x C x H x W
        N: Number of images, C: Number of channels, H: Height of the images, W: Width of the images
    epsilon : float
        Small value to avoid division by zero
    Returns
    -------
    Tensor
        Array of the SAM average indexes for each image of the N images
    """
    prod_scal = torch.sum(ms_gt * ms_fus, axis=1)
    norm_gt = torch.sum(ms_gt**2, axis=1)
    norm_fus = torch.sum(ms_fus**2, axis=1)
    lower_term = torch.sqrt(norm_gt * norm_fus)

    lower_term = torch.where(lower_term == 0, epsilon, lower_term) # to avoid division by zero
    SAM_map = torch.arccos(prod_scal/lower_term)

    angolo = torch.mean(SAM_map)
    SAM_index = torch.rad2deg(angolo)
    
    return SAM_index

def sam_map_torch(ms_gt:torch.Tensor, ms_fus:torch.Tensor, epsilon= 2 * 10**(-16)):
    """
    Parameters
    ----------
    ms_gt : Tensor 
        Batch of the ground truth multispectal images of shape N x C x H x W
        N: Number of images, C: Number of channels, H: Height of the images, W: Width of the images
    ms_fus : Tensor
        Batch of the fused multispectral images of shape N x C x H x W
        N: Number of images, C: Number of channels, H: Height of the images, W: Width of the images
    epsilon : float
        Small value to avoid division by zero
    Returns
    -------
    tuple[Tensor, Tensor]
        Array of the SAM average indexes for each image of the N images and H x W maps of the SAM indexes of the N images
    """
    prod_scal = torch.sum(ms_gt * ms_fus, axis=1)
    norm_gt = torch.sum(ms_gt**2, axis=1)
    norm_fus = torch.sum(ms_fus**2, axis=1)
    lower_term = torch.sqrt(norm_gt * norm_fus)
    
    lower_term = torch.where(lower_term == 0, epsilon, lower_term) # to avoid division by zero
    SAM_map = torch.arccos(prod_scal/lower_term)

    angolo = torch.mean(SAM_map)
    SAM_index = torch.rad2deg(angolo)

    return SAM_index, SAM_map
