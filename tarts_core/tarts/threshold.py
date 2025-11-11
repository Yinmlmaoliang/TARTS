#!/usr/bin/env python3
"""
Otsu Thresholding Methods for TARTS

Independent implementation of various Otsu thresholding algorithms
optimized for similarity score distributions in segmentation tasks.

Supports CUDA and torch.compile for efficient computation.
"""

from typing import Iterable, Optional

import torch
from torch.nn import functional as F


def histogram(image: torch.Tensor, bins=256, range: Optional[Iterable[float]] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculates the pixel value histogram for a grayscale image.
    While `torch.histogram` does not work with CUDA or `torch.compile`,
    this implementation does.

    Parameters:
        image: grayscale input image of shape (B, H, W) and float dtype.
        bins: number of bins used to calculate the image histogram.
        range: value range of the bins of form (min, max).

    Returns:
        (counts, bin_edges): Two tensors of shape (bins,)
    """
    if not range:
        range_min, range_max = image.min(), image.max()
    elif len(range) == 2:
        range_min, range_max = range[0], range[1]
    else:
        raise ValueError("range needs to be iterable of form: (min, max).")

    counts = torch.empty(bins, device=image.device, dtype=image.dtype)
    torch.histc(image, bins, min=range_min, max=range_max, out=counts)
    bin_edges = torch.linspace(range_min, range_max, bins, device=counts.device, dtype=counts.dtype)
    return counts, bin_edges


def _threshold_otsu_original(image: torch.Tensor, nbins=256) -> torch.Tensor:
    """Return threshold value based on Otsu's method.

    Parameters:
        image: grayscale input image of shape (B, H, W) and float dtype.
        nbins: number of bins used to calculate the image histogram.

    Returns:
        threshold: A threshold in [0,1] which can be used to binarize
        the grayscale image.

    References:
       [1]: https://en.wikipedia.org/wiki/Otsu's_Method
       [2]: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu

    Examples:
    >>> threshold = threshold_otsu(image)
    >>> binary = image <= threshold
    """
    counts, bin_edges = histogram(image, nbins, range=(0, 1))

    # class probabilities for all possible thresholds
    weight1 = torch.cumsum(counts, dim=0)
    weight2 = torch.cumsum(counts.flip(dims=(0,)), dim=0).flip(dims=(0,))

    # class means for all possible thresholds
    mean1 = torch.cumsum(counts * bin_edges, dim=0) / weight1
    mean2 = (
        torch.cumsum((counts * bin_edges).flip(dims=(0,)), dim=0)
        / weight2.flip(dims=(0,))
    ).flip(dims=(0,))

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = torch.argmax(variance12)
    threshold = idx.float() / nbins

    return threshold


def _threshold_otsu_standard(image: torch.Tensor, nbins=256) -> torch.Tensor:
    """Standard Otsu's method that properly handles similarity scores with correct thresholding."""
    counts, bin_edges = histogram(image, nbins, range=(0, 1))

    # Normalize counts to probabilities
    total = counts.sum()
    if total == 0:
        return torch.tensor(0.5, device=image.device)

    prob = counts / total

    # Cumulative sums
    prob_cumsum = torch.cumsum(prob, dim=0)

    # Class probabilities
    omega1 = prob_cumsum
    omega2 = 1.0 - omega1

    # Class means
    bin_centers = bin_edges

    # Cumulative intensity sums
    intensity_cumsum = torch.cumsum(prob * bin_centers, dim=0)
    total_mean = intensity_cumsum[-1]

    # Avoid division by zero
    eps = 1e-10
    mu1 = torch.zeros_like(omega1)
    mu2 = torch.zeros_like(omega2)

    # Only compute where omega > 0
    mask1 = omega1 > eps
    mask2 = omega2 > eps

    mu1[mask1] = intensity_cumsum[mask1] / omega1[mask1]
    mu2[mask2] = (total_mean - intensity_cumsum[mask2]) / omega2[mask2]

    # Between-class variance: σ²_B = ω₁ω₂(μ₁-μ₂)²
    # Only compute where both classes exist
    valid_mask = mask1 & mask2
    variance = torch.zeros_like(omega1)
    variance[valid_mask] = omega1[valid_mask] * omega2[valid_mask] * (mu1[valid_mask] - mu2[valid_mask]) ** 2

    # Find optimal threshold
    if variance.max() == 0:
        return torch.tensor(0.5, device=image.device)

    idx = torch.argmax(variance)
    threshold = bin_centers[idx]

    return threshold


def _threshold_valley_emphasis(image: torch.Tensor, nbins=256) -> torch.Tensor:
    """
    Return threshold value based on Valley-Emphasis method.

    This method weights the Otsu objective function with (1 - p(t))
    to favor threshold values with low probability (valley points).

    Reference: Ng, H. F. (2006). Automatic thresholding for defect detection.
    """
    counts, bin_edges = histogram(image, nbins, range=(0, 1))

    # Normalize to get probability distribution
    total_pixels = counts.sum()
    if total_pixels == 0:
        return torch.tensor(0.5, device=image.device)

    prob = counts / total_pixels

    # Class probabilities for all possible thresholds
    weight1 = torch.cumsum(prob, dim=0)
    weight2 = 1 - weight1

    # Class means using bin centers (not indices)
    bin_centers = bin_edges
    cumsum_prob_intensity = torch.cumsum(prob * bin_centers, dim=0)
    total_mean = cumsum_prob_intensity[-1]

    # Avoid division by zero
    eps = 1e-10
    mu1 = torch.zeros_like(weight1)
    mu2 = torch.zeros_like(weight2)

    mask1 = weight1 > eps
    mask2 = weight2 > eps

    mu1[mask1] = cumsum_prob_intensity[mask1] / weight1[mask1]
    mu2[mask2] = (total_mean - cumsum_prob_intensity[mask2]) / weight2[mask2]

    # Valley-emphasis weight: W(t) = 1 - p(t)
    valley_weight = 1 - prob

    # Between-class variance: σ²_B = ω₁ω₂(μ₁-μ₂)²
    valid_mask = mask1 & mask2
    variance = torch.zeros_like(weight1)
    variance[valid_mask] = weight1[valid_mask] * weight2[valid_mask] * (mu1[valid_mask] - mu2[valid_mask]) ** 2

    # Apply valley-emphasis weight
    weighted_variance = valley_weight * variance

    # Handle edge cases
    weighted_variance[0] = 0
    weighted_variance[-1] = 0

    if weighted_variance.max() == 0:
        return torch.tensor(0.5, device=image.device)

    idx = torch.argmax(weighted_variance)
    threshold = bin_centers[idx]
    return threshold


def _compute_valley_deepness(prob: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute valley deepness measure D(t) for each point in the histogram.

    D(t) = [lD(t) + rD(t)] / 2
    where lD(t) and rD(t) are left and right valley deepness measures.
    """
    nbins = prob.shape[0]
    device = prob.device
    dtype = prob.dtype

    # Apply Gaussian smoothing to reduce noise
    if sigma > 0:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        # Pad and convolve
        prob_padded = F.pad(prob.unsqueeze(0).unsqueeze(0),
                           (kernel_size//2, kernel_size//2), mode='replicate')
        prob_smooth = F.conv1d(prob_padded, kernel.unsqueeze(0).unsqueeze(0)).squeeze()
    else:
        prob_smooth = prob

    valley_deepness = torch.zeros(nbins, device=device, dtype=dtype)

    for t in range(nbins):
        # Left valley deepness
        left_deepness = 0
        if t > 0:
            left_diffs = prob_smooth[:t] - prob_smooth[t]
            left_diffs = torch.clamp(left_diffs, min=0)  # s() function
            if left_diffs.numel() > 0:
                left_deepness = left_diffs.max()

        # Right valley deepness
        right_deepness = 0
        if t < nbins - 1:
            right_diffs = prob_smooth[t+1:] - prob_smooth[t]
            right_diffs = torch.clamp(right_diffs, min=0)  # s() function
            if right_diffs.numel() > 0:
                right_deepness = right_diffs.max()

        # Overall valley deepness
        if left_deepness > 0 and right_deepness > 0:
            valley_deepness[t] = (left_deepness + right_deepness) / 2
        else:
            valley_deepness[t] = 0

    return valley_deepness


def _threshold_valley_deepness(image: torch.Tensor, nbins=256, sigma: float = 1.0) -> torch.Tensor:
    """
    Return threshold value based on the valley-deepness weighted method.

    This method weights the Otsu objective function with:
    W(t) = (1 - p(t)) + D(t)
    where D(t) is the valley deepness measure.

    Parameters:
        image: grayscale input image
        nbins: number of histogram bins
        sigma: standard deviation for Gaussian smoothing of histogram
    """
    counts, bin_edges = histogram(image, nbins, range=(0, 1))

    # Normalize to get probability distribution
    total_pixels = counts.sum()
    if total_pixels == 0:
        return torch.tensor(0.5, device=image.device)

    prob = counts / total_pixels

    # Class probabilities for all possible thresholds
    weight1 = torch.cumsum(prob, dim=0)
    weight2 = 1 - weight1

    # Class means using bin centers (not indices)
    bin_centers = bin_edges
    cumsum_prob_intensity = torch.cumsum(prob * bin_centers, dim=0)
    total_mean = cumsum_prob_intensity[-1]

    # Avoid division by zero
    eps = 1e-10
    mu1 = torch.zeros_like(weight1)
    mu2 = torch.zeros_like(weight2)

    mask1 = weight1 > eps
    mask2 = weight2 > eps

    mu1[mask1] = cumsum_prob_intensity[mask1] / weight1[mask1]
    mu2[mask2] = (total_mean - cumsum_prob_intensity[mask2]) / weight2[mask2]

    # Compute valley deepness
    valley_deepness = _compute_valley_deepness(prob, sigma)

    # Combined weight: W(t) = (1 - p(t)) + D(t)
    combined_weight = (1 - prob) + valley_deepness

    # Between-class variance: σ²_B = ω₁ω₂(μ₁-μ₂)²
    valid_mask = mask1 & mask2
    variance = torch.zeros_like(weight1)
    variance[valid_mask] = weight1[valid_mask] * weight2[valid_mask] * (mu1[valid_mask] - mu2[valid_mask]) ** 2

    # Apply combined weight
    weighted_variance = combined_weight * variance

    # Handle edge cases
    weighted_variance[0] = 0
    weighted_variance[-1] = 0

    if weighted_variance.max() == 0:
        return torch.tensor(0.5, device=image.device)

    idx = torch.argmax(weighted_variance)
    threshold = bin_centers[idx]
    return threshold


def threshold_otsu(image: torch.Tensor, nbins=256, method: str = 'standard', sigma: float = 1.0) -> torch.Tensor:
    """Return threshold value based on various Otsu methods.

    Parameters:
        image: grayscale input image of shape (B, H, W) and float dtype.
        nbins: number of bins used to calculate the image histogram.
        method: thresholding method ('standard', 'original', 'valley_emphasis', 'valley_deepness')
        sigma: standard deviation for Gaussian smoothing (only used for 'valley_deepness')

    Returns:
        threshold: A threshold in [0,1] which can be used to binarize
        the grayscale image.

    Available methods:
        - 'standard': Standard Otsu implementation (default, numerically stable)
        - 'original': Original implementation (kept for compatibility)
        - 'valley_emphasis': Valley-emphasis method favoring valley points
        - 'valley_deepness': Valley-deepness weighted method

    References:
       [1]: https://en.wikipedia.org/wiki/Otsu's_Method
       [2]: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu
       [3]: Ng, H. F. (2006). Automatic thresholding for defect detection.

    Examples:
    >>> # Default standard method (recommended)
    >>> threshold = threshold_otsu(image)
    >>> binary = image <= threshold
    >>>
    >>> # Use original method (for backward compatibility)
    >>> threshold = threshold_otsu(image, method='original')
    >>>
    >>> # Use valley emphasis method
    >>> threshold = threshold_otsu(image, method='valley_emphasis')
    >>>
    >>> # Use valley deepness method with custom smoothing
    >>> threshold = threshold_otsu(image, method='valley_deepness', sigma=2.0)
    """
    if method == 'original':
        return _threshold_otsu_original(image, nbins)
    elif method == 'standard':
        return _threshold_otsu_standard(image, nbins)
    elif method == 'valley_emphasis':
        return _threshold_valley_emphasis(image, nbins)
    elif method == 'valley_deepness':
        return _threshold_valley_deepness(image, nbins, sigma)
    else:
        raise ValueError(f"Unknown method '{method}'. Available methods: 'standard', 'original', 'valley_emphasis', 'valley_deepness'")
