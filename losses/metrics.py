import numpy as np

def dice_coefficient(y_true, y_pred , dim=None, smooth=1e-5):
    if dim is not None:
        y_true = y_true[...,dim]
        y_pred = y_pred[..., dim]
    intersection = np.sum(y_true * y_pred, axis=(1, 2))
    union = np.sum(y_true, axis=(1, 2)) + np.sum(y_pred, axis=(1, 2))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def jaccard_index(y_true, y_pred, dim=None, smooth=1e-5):
    if dim is not None:
        y_true = y_true[...,dim]
        y_pred = y_pred[..., dim]
    intersection = np.sum(y_true * y_pred, axis=(1, 2))
    union = np.sum(y_true, axis=(1, 2)) + np.sum(y_pred, axis=(1, 2)) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    return jaccard