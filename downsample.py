import numpy as np
import torch
import math
# from skimage import transform

from typing import Literal

class MTF():
    def __init__(self, sensor: str, channels: int, device: str, ratio = 4 ,kernel_size=41):
        """
        Parameters
        ----------
        sensor : str
            Sensor name (qb, ikonos, geoeye1, wv2, wv3, wv4)
        channels : int
            Number of channels in the MS image
        device : str
            Device to run the model on (cpu, cuda)
        ratio : int
            Downscaling factor PAN/MS(default is 4) 
        kernel_size : int
            Size of the kernel (default is 41)
        """
        self.ratio = ratio
        self.sensor = sensor.lower()
        self.channels = channels
        self.kernel_size = kernel_size
        self.device = device

        GNyq_dict_ms_default = [0.3 for _ in range(self.channels)]
        GNyq_dict_ms = {
            'qb':       [0.34, 0.32, 0.30, 0.22], # Band Order: B,G,R,NIR
            'ikonos':   [0.26, 0.28, 0.29, 0.28],    # Band Order: B,G,R,NIR
            'geoeye1':  [0.23, 0.23, 0.23, 0.23],    # Band Order: B,G,R,NIR
            'wv2':      [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.27],
            'wv3':      [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315],
            'wv4':      [0.23, 0.23, 0.23, 0.23] # Band Order: B, G, R, NIR
        } 
        self.GNyq_dict_ms = GNyq_dict_ms.get(self.sensor, GNyq_dict_ms_default)
        self.GNyq_dict_ms = np.asarray(self.GNyq_dict_ms, dtype=np.float32)

        GNyq_dict_pan_default = 0.15
        GNyq_dict_pan = {
            'qb':       0.15,
            'ikonos':   0.17,
            'geoeye1':  0.16,
            'wv2':      0.11,
            'wv3':      0.5,
            'wv4':      0.16,
        }
        self.GNyq_dict_pan = GNyq_dict_pan.get(self.sensor, GNyq_dict_pan_default)
        self.GNyq_dict_pan = np.asarray([self.GNyq_dict_pan], dtype=np.float32)

        self.kernel_ms = NyquistFilterGenerator(self.GNyq_dict_ms, self.ratio, kernel_size)
        self.kernel_ms_torch = mtf_kernel_to_torch(self.kernel_ms).to(self.device)

        self.kernel_pan = NyquistFilterGenerator(self.GNyq_dict_pan, self.ratio, kernel_size)
        self.kernel_pan_torch = mtf_kernel_to_torch(self.kernel_pan).to(self.device)

    def genMTF_ms_np(self, ms: np.ndarray):
        """
        Parameters
        ----------
        ms : np.ndarray
            MS image of shape H x W x C
            H: Height of the image, W: Width of the image, C: Number of channels
        Returns
        -------
        np.ndarray
            MS image after MTF
        """
        ms = img_to_torch(ms, "ms")
        conved_ms = depthConv(self.kernel_ms_torch, ms, self.channels, self.kernel_size, self.ratio)
        conved_ms = img_to_numpy(conved_ms, "ms")
        # MS_scale = (math.floor(conved_ms.shape[0] / self.ratio), math.floor(conved_ms.shape[1] / self.ratio), conved_ms.shape[2])
        # I_MS_LR = transform.resize(conved_ms, MS_scale, order=0)
        return conved_ms
        
    def genMTF_pan_np(self, pan: np.ndarray):
        """
        Parameters
        ----------
        pan : np.ndarray
            PAN image of shape H x W
            H: Height of the image, W: Width of the image
        Returns
        -------
        np.ndarray
            PAN image after MTF
        """
        pan = img_to_torch(pan, "pan")
        conved_pan = depthConv(self.kernel_pan_torch, pan, 1, self.kernel_size, self.ratio)
        conved_pan = img_to_numpy(conved_pan, "pan")
        # PAN_scale = (math.floor(conved_pan.shape[0] / self.ratio), math.floor(conved_pan.shape[1] / self.ratio))
        # I_PAN_LR = transform.resize(conved_pan, PAN_scale, order=0)
        return conved_pan
    
    def genMTF_ms_torch(self, ms_batch:torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ms_batch : torch.Tensor
            MS image of shape B x C x H x W
            B: Batch size, C: Number of channels, H: Height of the image, W: Width of the image
        Returns
        ----------
        torch.Tensor
            MS image after MTF
        """
        return depthConv(self.kernel_ms_torch, ms_batch, self.channels, self.kernel_size, self.ratio)

    def genMTF_pan_torch(self, pan_batch:torch.Tensor) -> torch.Tensor: 
        """
        Parameters
        pan_batch : torch.Tensor
            PAN image of shape B x 1 x H x W
            B: Batch size, H: Height of the image, W: Width of the image
        Returns
        -------
        torch.Tensor
            PAN image after MTF
        """
        return depthConv(self.kernel_pan_torch, pan_batch, 1, self.kernel_size, self.ratio)

def depthConv(mtf_kernel: torch.Tensor, img: torch.Tensor, channels: int, kernel_size= 41, ratio = 4):
    conv = torch.nn.Conv2d(in_channels=channels,
                            out_channels=channels, 
                            padding=math.ceil(kernel_size / 2),
                            kernel_size=mtf_kernel.shape, 
                            groups=channels, 
                            bias=False, 
                            padding_mode='replicate')
    conv.weight.data = mtf_kernel
    conv.weight.requires_grad = False
    conved = conv(img)
    conved = torch.nn.functional.interpolate(conved, scale_factor= 1/ratio, mode="bicubic")
    return conved

def img_to_numpy(img: torch.Tensor, method: Literal["ms", "pan"]) -> np.ndarray:
    img = img.numpy()
    img = np.squeeze(img)
    if method == "ms":
        img = np.moveaxis(img, 0, -1)
    return img

def img_to_torch(img: np.ndarray, method: Literal["ms", "pan"]) -> torch.Tensor:
    if method == "ms":
        img = np.moveaxis(img, -1, 0)
        img = np.expand_dims(img, axis=0)
    else:       
        img = np.expand_dims(img, [0, 1])
    img = torch.from_numpy(img.astype(np.float32))
    return img

def mtf_kernel_to_torch(kernel: np.ndarray) -> torch.Tensor:
    kernel = np.moveaxis(kernel, -1, 0)
    kernel = np.expand_dims(kernel, axis=1)
    kernel = kernel.astype(np.float32)
    kernel = torch.from_numpy(kernel).type(torch.float32)
    return kernel

def fspecial_gauss(size, sigma):
    # Function to mimic the 'fspecial' gaussian MATLAB function
    m, n = [(ss-1.)/2. for ss in size]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def fir_filter_wind(Hd, w):
    """
    compute fir filter with window method
    Hd:     desired freqeuncy response (2D)
    w:      window (2D)
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = np.clip(h, a_min=0, a_max=np.max(h))
    h = h / np.sum(h)
    
    return h

def NyquistFilterGenerator(Gnyq, ratio, N):
    assert isinstance(Gnyq, (np.ndarray, list)), 'Error: GNyq must be a list or a ndarray'
    if isinstance(Gnyq, list):
        Gnyq = np.asarray(Gnyq)
    nbands = Gnyq.shape[0]

    kernel = np.zeros((N, N, nbands))  # generic kerenel (for normalization purpose)
    fcut = 1 / np.double(ratio)
    for j in range(nbands):
        alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(Gnyq[j])))
        H = fspecial_gauss((N,N), alpha)
        Hd = H / np.max(H)
        h = np.kaiser(N, 0.5)
        kernel[:, :, j] = np.real(fir_filter_wind(Hd, h))
    return kernel

def interp23tap_GPU(img, ratio):

    assert((2**(round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'

    r,c,b = img.shape

    CDF23 = np.asarray([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0, -0.000060081482])
    CDF23 = [element * 2 for element in CDF23]
    BaseCoeff = np.expand_dims(np.concatenate([np.flip(CDF23[1:]), CDF23]), axis=-1)
    BaseCoeff = np.expand_dims(BaseCoeff, axis=(0,1))
    BaseCoeff = np.concatenate([BaseCoeff]*b, axis=0)


    BaseCoeff = torch.from_numpy(BaseCoeff)
    img = img.astype(np.float32)
    img = np.moveaxis(img, -1, 0)


    for z in range(int(ratio/2)):

        I1LRU = np.zeros((b, (2 ** (z+1)) * r, (2 ** (z+1)) * c))

        if z == 0:
            I1LRU[:,1::2, 1::2] = img
        else:
            I1LRU [:,::2,::2] = img

        I1LRU = np.expand_dims(I1LRU, axis=0)
        conv = torch.nn.Conv2d(in_channels=b, out_channels=b, padding=(11,0),
                            kernel_size=BaseCoeff.shape, groups=b, bias=False, padding_mode='circular')

        conv.weight.data = BaseCoeff
        conv.weight.requires_grad = False

        t = conv(torch.transpose(torch.from_numpy(I1LRU), 2, 3))
        img = conv(torch.transpose(t, 2,3)).numpy()
        img = np.squeeze(img)

    img = np.moveaxis(img, 0,-1)


    return img