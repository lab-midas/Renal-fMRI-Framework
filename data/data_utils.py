"""
Utility functions for multi-parametric renal MRI preprocessing and analysis.

This module provides core utility functions for:
- Image normalization and contrast stretching
- 3D cropping and resizing operations
- Image denoising using total variation
- Mask conversion utilities
- Image registration utilities (to be added)

All functions are designed to work with numpy arrays representing medical images,
typically with dimensions (height, width, depth) or (height, width, depth, channels).
"""

import numpy as np
from skimage import exposure
from skimage.restoration import denoise_tv_bregman as denoise_tv


def contrastStretch(ipImg, ipMask, lwr_prctile=10, upr_prctile=100):
    """
    Apply histogram-based contrast stretching to an image.

    This function rescales the intensity range of the input image to improve contrast.
    If a mask is provided, percentiles are computed only within the masked region.

    Args:
        ipImg: Input image array (can be 2D, 3D, or 4D)
        ipMask: Optional binary mask to restrict percentile calculation.
              Should have the same spatial dimensions as image.
        lwr_prctile: Lower percentile for contrast stretching (default: 10.0)
        upr_prctile: Upper percentile for contrast stretching (default: 100.0)

    Returns:
        Contrast-stretched image with same shape as input

    Example:
        >>> img = np.random.rand(256, 256, 10)
        >>> mask = np.random.choice([0, 1], size=(256, 256, 10))
        >>> stretched = contrastStretch(img, mask, 5, 95)
    """
    mm = ipImg[ipMask > 0]
    p10 = np.percentile(mm, lwr_prctile)
    p100 = np.percentile(mm, upr_prctile)
    opImg = exposure.rescale_intensity(ipImg, in_range=(p10, p100))
    return opImg


def myCrop3D(ipImg, opShape):
    ''' Crop a 3D volume (H x W x D) to the following size (opShape x opShape x D)'''
    xDim, yDim = opShape
    zDim = ipImg.shape[2]
    opImg = np.zeros((xDim, yDim, zDim))

    xPad = xDim - ipImg.shape[0]
    yPad = yDim - ipImg.shape[1]

    x_lwr = int(np.floor(np.abs(xPad) / 2))
    x_upr = int(np.ceil(np.abs(xPad) / 2))
    y_lwr = int(np.floor(np.abs(yPad) / 2))
    y_upr = int(np.ceil(np.abs(yPad) / 2))
    if xPad >= 0 and yPad >= 0:
        opImg[x_lwr:xDim - x_upr, y_lwr:yDim - y_upr, ...] = ipImg
    elif xPad < 0 and yPad < 0:
        xPad = np.abs(xPad)
        yPad = np.abs(yPad)
        opImg = ipImg[x_lwr: -x_upr, y_lwr:- y_upr, :]
    elif xPad < 0 and yPad >= 0:
        xPad = np.abs(xPad)
        temp_opImg = ipImg[x_lwr: -x_upr, :, :]
        opImg[:, y_lwr:yDim - y_upr, :] = temp_opImg
    else:
        yPad = np.abs(yPad)
        temp_opImg = ipImg[:, y_lwr: -y_upr, :]
        opImg[x_lwr:xDim - x_upr, :, :] = temp_opImg
    return opImg


def crop_or_pad_3d(
        image,
        target_shape,
        mode= 'center'
) -> np.ndarray:
    """
    Crop or pad a 3D volume to a target spatial shape.

    This function handles both cropping (if image is larger than target)
    and padding (if image is smaller than target). The operation is applied
    to the spatial dimensions (first 2 or 3 dimensions) while preserving
    any additional dimensions (like channels).

    The function centers the image in the target space, so cropping/padding
    is applied symmetrically when possible.

    Args:
        image: Input array. Can be:
            - 3D: (height, width, depth)
            - 4D: (height, width, depth, channels)
            - 2D: (height, width) - will be treated as (height, width, 1)
        target_shape: Target spatial shape. Can be:
            - (height, width) for 2D cropping/padding
            - (height, width, depth) for 3D cropping/padding
        mode: Padding/cropping strategy. Currently only 'center' is supported.
              'center' centers the image in the target space.

    Returns:
        Cropped/padded array with spatial dimensions matching target_shape.
        The output will have the same number of dimensions as the input
        (3D or 4D), with spatial dimensions adjusted to target_shape.

    Raises:
        ValueError: If target_shape has invalid length (not 2 or 3)
        ValueError: If mode is not 'center'
        ValueError: If image has unsupported number of dimensions

    Examples:
        >>> # Pad a small 3D image to 256x256x20
        >>> img = np.random.rand(200, 200, 15)
        >>> padded = crop_or_pad_3d(img, (256, 256, 20))
        >>> print(padded.shape)
        (256, 256, 20)

        >>> # Crop a large 4D image to 256x256 while preserving depth and channels
        >>> img = np.random.rand(300, 300, 30, 3)
        >>> cropped = crop_or_pad_3d(img, (256, 256))
        >>> print(cropped.shape)
        (256, 256, 30, 3)

        >>> # Center padding maintains content in the middle
        >>> img = np.random.rand(100, 100)
        >>> padded = crop_or_pad_3d(img, (256, 256))
        >>> # Content will be centered with 78 pixels padding on each side
    """
    import numpy as np

    # Validate inputs
    if mode != 'center':
        raise ValueError(f"Unsupported mode: '{mode}'. Only 'center' is currently supported.")

    # Handle different target_shape formats
    if len(target_shape) == 2:
        target_h, target_w = target_shape
        # Preserve original depth
        if image.ndim >= 3:
            target_d = image.shape[2]
        else:
            target_d = 1
    elif len(target_shape) == 3:
        target_h, target_w, target_d = target_shape
    else:
        raise ValueError(
            f"target_shape must have length 2 or 3, got {len(target_shape)}. "
            "For 2D operations use (H, W), for 3D use (H, W, D)."
        )

    # Get current dimensions based on input dimensionality
    orig_ndim = image.ndim

    if orig_ndim == 2:
        # 2D input: (H, W)
        orig_h, orig_w = image.shape
        orig_d = 1
        has_channels = False
    elif orig_ndim == 3:
        # 3D input: (H, W, D)
        orig_h, orig_w, orig_d = image.shape
        has_channels = False
    elif orig_ndim == 4:
        # 4D input: (H, W, D, C)
        orig_h, orig_w, orig_d, num_channels = image.shape
        has_channels = True
    else:
        raise ValueError(
            f"Image must be 2D, 3D, or 4D, got {orig_ndim}D with shape {image.shape}"
        )

    # Initialize output array
    if has_channels:
        output = np.zeros((target_h, target_w, target_d, num_channels), dtype=image.dtype)
    elif orig_ndim == 3 or (orig_ndim == 2 and target_d > 1):
        output = np.zeros((target_h, target_w, target_d), dtype=image.dtype)
    else:  # 2D output
        output = np.zeros((target_h, target_w), dtype=image.dtype)

    # Calculate cropping/padding indices for target space
    # These determine where in the output array we'll place the original data
    h_start_out = max(0, (target_h - orig_h) // 2)
    h_end_out = min(target_h, h_start_out + orig_h)
    w_start_out = max(0, (target_w - orig_w) // 2)
    w_end_out = min(target_w, w_start_out + orig_w)
    d_start_out = max(0, (target_d - orig_d) // 2)
    d_end_out = min(target_d, d_start_out + orig_d)

    # Calculate source indices (where to take data from original image)
    # These handle the case when we need to crop (image larger than target)
    h_start_in = max(0, (orig_h - target_h) // 2)
    h_end_in = h_start_in + (h_end_out - h_start_out)
    w_start_in = max(0, (orig_w - target_w) // 2)
    w_end_in = w_start_in + (w_end_out - w_start_out)
    d_start_in = max(0, (orig_d - target_d) // 2)
    d_end_in = d_start_in + (d_end_out - d_start_out)

    # Copy data based on dimensionality
    try:
        if has_channels:
            # 4D case: (H, W, D, C)
            output[h_start_out:h_end_out,
            w_start_out:w_end_out,
            d_start_out:d_end_out, :] = \
                image[h_start_in:h_end_in,
                w_start_in:w_end_in,
                d_start_in:d_end_in, :]
        elif orig_ndim == 3 or (orig_ndim == 2 and target_d > 1):
            # 3D case: (H, W, D)
            output[h_start_out:h_end_out,
            w_start_out:w_end_out,
            d_start_out:d_end_out] = \
                image[h_start_in:h_end_in,
                w_start_in:w_end_in,
                d_start_in:d_end_in]
        else:
            # 2D case: (H, W)
            # Note: For 2D, we ignore depth dimensions
            output[h_start_out:h_end_out, w_start_out:w_end_out] = \
                image[h_start_in:h_end_in, w_start_in:w_end_in]

    except Exception as e:
        # Provide detailed error message for debugging
        raise RuntimeError(
            f"Error during crop/pad operation:\n"
            f"  Image shape: {image.shape}\n"
            f"  Target shape: ({target_h}, {target_w}, {target_d})\n"
            f"  Output indices: h[{h_start_out}:{h_end_out}], "
            f"w[{w_start_out}:{w_end_out}], d[{d_start_out}:{d_end_out}]\n"
            f"  Input indices: h[{h_start_in}:{h_end_in}], "
            f"w[{w_start_in}:{w_end_in}], d[{d_start_in}:{d_end_in}]\n"
            f"  Error: {e}"
        ) from e

    return output


def normalize(img):
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:  # Avoid division by zero
        return (img - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(img)


def performDenoising(ipImg, wts):
    """
        Apply total variation denoising (Bregman algorithm) to an image.

        Total variation denoising removes noise while preserving edges.
        The image is temporarily scaled to [0, 1] for denoising and then
        rescaled back to original intensity range.

        Args:
            image: Input image array (2D or 3D)
            weight: Denoising weight. Higher values = more smoothing (default: 30.0)
            eps: Tolerance for convergence (default: 1e-4)
            max_iter: Maximum number of iterations (default: 200)

        Returns:
            Denoised image with same shape and intensity range as input

        Example:
            >>> noisy = np.random.rand(256, 256) + 0.1 * np.random.randn(256, 256)
            >>> denoised = performDenoising(noisy, weight=20)
        """
    max_val = np.max(ipImg)
    ipImg = ipImg / max_val  # Rescale to 0 to 1
    opImg = denoise_tv(ipImg, wts)
    opImg = opImg * max_val
    return opImg

def mask_to_one_hot(mask: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """
    Convert a grayscale mask to a one-hot encoded format.

    Args:
        mask (np.ndarray): 2D numpy array where each pixel corresponds to a class index.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: One-hot encoded mask with shape (H, W, num_classes).
    """
    one_hot = np.zeros((*mask.shape, num_classes), dtype=np.float32)
    for class_idx in range(num_classes):
        one_hot[..., class_idx] = (mask == class_idx).astype(np.float32)
    return one_hot


def transpose_array(arr):
    if arr.ndim == 3:
        return np.transpose(arr, (2,0,1))
    elif arr.ndim == 4:
        return np.transpose(arr, (2, 0,1, 3))
    else:
        raise ValueError("Input array must be 3D or 4D.")