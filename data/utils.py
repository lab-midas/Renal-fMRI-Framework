import numpy as np
from skimage.restoration import denoise_tv_bregman as denoise_tv


def contrastStretch(ipImg, ipMask, lwr_prctile=10, upr_prctile=100):
    ''' Histogram based contrast stretching '''
    from skimage import exposure
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


def normalize_img_zmean(img, mask):
    ''' Zero mean unit standard deviation normalization based on a mask'''
    mask_signal = img[mask > 0]
    mean_ = mask_signal.mean()
    std_ = mask_signal.std()
    img = (img - mean_) / std_
    return img


def normalize(img):
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:  # Avoid division by zero
        return (img - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(img)


def performDenoising(ipImg, wts):
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