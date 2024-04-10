import numpy as np
import torch
import math
import skimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def mse(x, y):
    return np.mean(np.abs(x - y)**2)


def psnr(x, y):
    '''
    Measures the PSNR of recon w.r.t x.
    Image must be of either integer (0, 256) or float value (0,1)
    :param x: [m,n]
    :param y: [m,n]
    :return:
    '''
    assert x.shape == y.shape
    assert x.dtype == y.dtype or np.issubdtype(x.dtype, np.float) \
        and np.issubdtype(y.dtype, np.float)
    if x.dtype == np.uint8:
        max_intensity = 256
    else:
        max_intensity = 1

    mse = np.sum((x - y) ** 2).astype(float) / x.size
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)


def complex_psnr(x, y, peak='normalized'):
    '''
    x: reference image
    y: reconstructed image
    peak: normalised or max
    Notice that ``abs'' squares
    Be careful with the order, since peak intensity is taken from the reference
    image (taking from reconstruction yields a different value).
    '''
    # a = np.abs(x - y) ** 2
    mse = np.mean(np.abs(x - y)**2)
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(x))**2/mse)
    else:
        return 10*np.log10(1./mse + 1e-5)

def nrmse(outputs, targets):
    """
    Normalized root-mean square error
    :param outputs: Module's outputs
    :param targets: Target signal to be learned
    :return: Normalized root-mean square deviation
    """
    # Flatten tensors
    outputs = outputs.reshape(-1)
    targets = targets.reshape(-1)

    # Check dim
    # if outputs.size() != targets.size():
    #     raise ValueError(u"Ouputs and targets tensors don have the same number of elements")
    # end if

    # Normalization with N-1
    var = np.std(targets) ** 2

    # Error
    error = (targets - outputs) ** 2

    # Return
    return float(math.sqrt(np.mean(error) / var))

def complex_ssim(gt, pred, maxval=None):
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    ssim = ssim / gt.shape[0]

    return ssim