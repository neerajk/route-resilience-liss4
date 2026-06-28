"""Sensor-realistic resolution degradation (MTF/Gaussian blur-downsample).

WHY (the DeepGlobe-vs-LISS-IV problem, MASTER_PLAN §1.3): a real coarse sensor
INTEGRATES radiance over its larger ground cell through its Point-Spread Function
(PSF), then samples. Naive decimation skips the integration and leaves aliasing
(violates Nyquist). The standard, sensor-realistic recipe is the
**blur-downsample (BD) model**: convolve with an MTF-derived Gaussian, THEN
resample. We use it for two purposes:
  1. degrade 0.5 m DeepGlobe -> ~5.8 m before pretraining (domain/scale match);
  2. scale augmentation (random GSD) for scale-robust training.

The MTF (Modulation Transfer Function) measures how a sensor attenuates spatial
frequencies; PSF = inverse Fourier transform of the MTF, well-approximated by a
Gaussian for the composite blur.

References
----------
- Blur-downsample with MTF-based Gaussian filters (standard degradation model):
  "High-Resolution Satellite Image Super-Resolution Using Image Degradation Model
   with MTF-Based Filters", Korean J. Remote Sensing (2023).
  https://www.kjrs.org/journal/view.html?pn=search&uid=841&vmd=Full
- Realistic degradation for training: "Super-Resolving Beyond Satellite Hardware
  Using Realistically Degraded Images" (arXiv:2103.06270).
- Gaussian PSF approximation of composite sensor blur (widely used in EO SR).

Pure NumPy + scikit-image -> runs on the stock interpreter TODAY.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import gaussian
from skimage.transform import resize


def mtf_to_sigma(scale_factor: float, mtf_at_nyquist: float = 0.2) -> float:
    """Gaussian sigma (in HR pixels) for a given MTF value at the LR Nyquist freq.

    Derivation: a Gaussian PSF has MTF(f) = exp(-2 (pi sigma f)^2). The LR Nyquist
    frequency is f_n = 1/(2*scale) cycles per HR pixel. Solving
    MTF(f_n) = mtf_at_nyquist for sigma:
        sigma = scale/pi * sqrt( -ln(mtf_at_nyquist) / 2 )
    Typical sensor MTF at Nyquist ~0.2 (e.g., many spaceborne imagers target
    ~0.2-0.3). Smaller mtf_at_nyquist => stronger blur.
    """
    if not (0 < mtf_at_nyquist < 1):
        raise ValueError("mtf_at_nyquist must be in (0,1)")
    return (scale_factor / np.pi) * np.sqrt(-np.log(mtf_at_nyquist) / 2.0)


def blur_downsample(image: np.ndarray, scale_factor: float,
                    mtf_at_nyquist: float = 0.2,
                    channel_axis: int | None = -1) -> np.ndarray:
    """Degrade an HR image to 1/scale_factor resolution via the BD model.

    Parameters
    ----------
    image : HR array. With channel_axis set, that axis is treated as channels.
    scale_factor : e.g. 5.8/0.5 = 11.6 to go from 0.5 m -> 5.8 m.
    mtf_at_nyquist : sensor sharpness at Nyquist (lower => more blur).

    Returns the degraded (smaller) array.
    """
    if scale_factor <= 1:
        return image
    sigma = mtf_to_sigma(scale_factor, mtf_at_nyquist)
    blurred = gaussian(image, sigma=sigma, channel_axis=channel_axis,
                       preserve_range=True)
    if channel_axis is None:
        out_shape = tuple(max(1, int(round(s / scale_factor))) for s in image.shape)
    else:
        ax = channel_axis % image.ndim
        out_shape = list(image.shape)
        for i in range(image.ndim):
            if i != ax:
                out_shape[i] = max(1, int(round(image.shape[i] / scale_factor)))
        out_shape = tuple(out_shape)
    return resize(blurred, out_shape, order=1, preserve_range=True,
                  anti_aliasing=False).astype(image.dtype)


def degrade_then_restore(image: np.ndarray, scale_factor: float,
                         mtf_at_nyquist: float = 0.2,
                         channel_axis: int | None = -1) -> np.ndarray:
    """Degrade then upsample back to the ORIGINAL size.

    Used for SCALE AUGMENTATION: the image keeps its pixel dimensions but loses
    fine detail as if seen by a coarser sensor — teaching the model to be
    robust across GSDs (the scale-robustness angle, cf. Scale-MAE, Reed et al.,
    ICCV 2023).
    """
    small = blur_downsample(image, scale_factor, mtf_at_nyquist, channel_axis)
    return resize(small, image.shape, order=1, preserve_range=True,
                  anti_aliasing=False).astype(image.dtype)
