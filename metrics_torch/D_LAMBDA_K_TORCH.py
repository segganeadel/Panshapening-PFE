import torch
from .Q2N_TORCH import q2n_torch

def d_lambda_k_torch(ms_fus:torch.Tensor, ms_orig_low:torch.Tensor) -> torch.Tensor:
    """
    TODO :
    """
    low_fus = torch.nn.functional.interpolate(ms_fus, ms_orig_low.shape[2:])
    q2n_index = q2n_torch(low_fus, ms_orig_low)
    d_lambda_k_index = 1 - q2n_index
    
    return d_lambda_k_index