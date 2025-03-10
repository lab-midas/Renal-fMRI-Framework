# core python

import neurite as ne
# third party
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class Grad:
    """
    N-D gradient loss.
    loss_mult can be used to scale the loss value - this is recommended if
    the gradient is computed on a downsampled vector field (where loss_mult
    is equal to the downsample factor).
    """

    def __init__(self, penalty='l1', loss_mult=None, vox_weight=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.vox_weight = vox_weight

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            yp = K.permute_dimensions(y, r)
            dfi = yp[1:, ...] - yp[:-1, ...]

            if self.vox_weight is not None:
                w = K.permute_dimensions(self.vox_weight, r)
                # TODO: Need to add square root, since for non-0/1 weights this is bad.
                dfi = w[1:, ...] * dfi

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss_bd(self, img, y_pred):
        """
        returns Tensor of size [bs]
        """

        if self.penalty == 'l1':
            dif = [tf.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        img_dif = [tf.abs(f) for f in self._diffs(img)]
        weights = [tf.exp(-tf.reduce_mean(tf.abs(im), axis=1, keepdims=True) * 10) for im in img_dif] # pixels=10
        dif = [weights[i] * dif[i] for i in range(len(dif))]

        df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
        grad = tf.add_n(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad

    def loss(self, _, y_pred):
        """
        returns Tensor of size [bs]
        """

        if self.penalty == 'l1':
            dif = [tf.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
        grad = tf.add_n(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad

    def mean_loss(self, y_true, y_pred):
        """
        returns Tensor of size ()
        """

        return K.mean(self.loss(y_true, y_pred))


class GradReshape:
    """
    N-D gradient loss.
    loss_mult can be used to scale the loss value - this is recommended if
    the gradient is computed on a downsampled vector field (where loss_mult
    is equal to the downsample factor).
    """

    def __init__(self, penalty='l1', loss_mult=None, vox_weight=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.vox_weight = vox_weight

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            yp = K.permute_dimensions(y, r)
            dfi = yp[1:, ...] - yp[:-1, ...]

            if self.vox_weight is not None:
                w = K.permute_dimensions(self.vox_weight, r)
                # TODO: Need to add square root, since for non-0/1 weights this is bad.
                dfi = w[1:, ...] * dfi

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss(self, _, y_pred):
        """
        returns Tensor of size [bs]
        """
        mm, nn, cc = y_pred.shape[2:]
        y_pred = tf.reshape(y_pred, [-1, mm, nn, cc])
        #print(y_pred.shape)

        if self.penalty == 'l1':
            dif = [tf.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
        grad = tf.add_n(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad

    def mean_loss(self, y_true, y_pred):
        """
        returns Tensor of size ()
        """

        return K.mean(self.loss(y_true, y_pred))





class GradMax:
    """
    N-D gradient loss.
    loss_mult can be used to scale the loss value - this is recommended if
    the gradient is computed on a downsampled vector field (where loss_mult
    is equal to the downsample factor).
    """

    def __init__(self, penalty='l1', loss_mult=None, vox_weight=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.vox_weight = vox_weight

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            yp = K.permute_dimensions(y, r)
            dfi = yp[1:, ...] - yp[:-1, ...]

            if self.vox_weight is not None:
                w = K.permute_dimensions(self.vox_weight, r)
                # TODO: Need to add square root, since for non-0/1 weights this is bad.
                dfi = w[1:, ...] * dfi

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss(self, _, y_pred):
        """
        returns Tensor of size [bs]
        """

        if self.penalty == 'l1':
            dif = [tf.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
        grad = tf.add_n(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        max_loss = 0.1 * tf.reduce_mean(tf.square(y_pred))

        grad = grad + max_loss

        return grad

    def mean_loss(self, y_true, y_pred):
        """
        returns Tensor of size ()
        """

        return K.mean(self.loss(y_true, y_pred))


class MutualInformation(ne.metrics.MutualInformation):
    """
    Soft Mutual Information approximation for intensity volumes

    More information/citation:
    - Courtney K Guo.
      Multi-modal image registration with unsupervised deep learning.
      PhD thesis, Massachusetts Institute of Technology, 2019.
    - M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
      SynthMorph: learning contrast-invariant registration without acquired images
      IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
      https://doi.org/10.1109/TMI.2021.3116879
    """

    def loss(self, y_true, y_pred):
        return -self.volumes(y_true, y_pred)

class mse_from_difference_loss:

    def loss(self, _, y_pred):
        loss = tf.reduce_mean(tf.square(y_pred))
        return loss


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(y_pred.get_shape().as_list()) - 2
        vol_axes = list(range(1, ndims + 1))

        top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
        bottom = tf.reduce_sum(y_true + y_pred, vol_axes)

        div_no_nan = tf.math.divide_no_nan if hasattr(tf.math, 'divide_no_nan') else tf.div_no_nan  # pylint: disable=no-member
        dice = tf.reduce_mean(div_no_nan(top, bottom))
        return 1-dice

class MutualInformation2:
    def loss(self,y_true, y_pred):
        bin_centers = np.linspace(0, 1, 100) # return specified interval numbers

        sigma_ratio = 1

        vol_bin_centers = K.variable(bin_centers)
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        preterm = K.variable(1 / (2 * np.square(sigma)))

        y_pred = K.clip(y_pred, 0, 1)
        y_true = K.clip(y_true, 0, 1)


        # reshape: flatten images into shape (batch_size, heightxwidthxdepthxchan, 1)
        y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
        y_true = K.expand_dims(y_true, 2)
        y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
        y_pred = K.expand_dims(y_pred, 2)

        nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, np.prod(vol_bin_centers.get_shape().as_list())]
        vbc = K.reshape(vol_bin_centers, o)

        # compute image terms
        I_a = K.exp(- preterm * K.square(y_true - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- preterm * K.square(y_pred - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a, (0, 2, 1))
        pab = K.batch_dot(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
        pab /= nb_voxels
        pa = tf.reduce_mean(I_a, 1, keepdims=True)
        pb = tf.reduce_mean(I_b, 1, keepdims=True)

        papb = K.batch_dot(K.permute_dimensions(pa, (0, 2, 1)), pb) + K.epsilon()
        mi = K.sum(K.sum(pab * K.log(pab / papb + K.epsilon()), 1), 1)
        return - mi



# https://github.com/SZUHvern/TMI_multi-contrast-registration/blob/main/registration.py

class mi:
    def __init__(self, num_bins=100, sigma_ratio=1):
        super(mi, self).__init__()
        self.num_bins = num_bins
        self.sigma_ratio = sigma_ratio
        self.vol_bin_centers = K.variable(np.linspace(0, 1, num_bins))
        self.sigma = np.mean(np.diff(self.vol_bin_centers)) * sigma_ratio
        self.preterm = K.variable(1 / (2 * np.square(self.sigma)))

    def loss(self, y_true, y_pred):
        y_pred = K.clip(y_pred, 0, 1)
        y_true = K.clip(y_true, 0, 1)
        y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
        y_true = K.expand_dims(y_true, 2)
        y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
        y_pred = K.expand_dims(y_pred, 2)
        nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

        o = [1, 1, np.prod(self.vol_bin_centers.shape.as_list())]
        vbc = K.reshape(self.vol_bin_centers, o)

        I_a = K.exp(- self.preterm * K.square(y_true - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- self.preterm * K.square(y_pred - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        I_a_permute = K.permute_dimensions(I_a, (0, 2, 1))
        pab = K.batch_dot(I_a_permute, I_b)
        pab /= nb_voxels
        pa = tf.reduce_mean(I_a, 1, keepdims=True)
        pb = tf.reduce_mean(I_b, 1, keepdims=True)

        papb = K.batch_dot(K.permute_dimensions(pa, (0, 2, 1)), pb) + K.epsilon()
        mi = K.sum(K.sum(pab * K.log(pab / papb + K.epsilon()), 1), 1)
        return -mi


class smooth:
    """
    N-D gradient loss.
    loss_mult can be used to scale the loss value - this is recommended if
    the gradient is computed on a downsampled vector field (where loss_mult
    is equal to the downsample factor).
    """

    def __init__(self, penalty='l1', loss_mult=None, vox_weight=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.vox_weight = vox_weight

    def gradient(self, var):
        grad_var_nor = tf.pad(var, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        grad_var_1 = tf.pad(var, [[0, 0], [2, 0], [1, 1], [0, 0]], "CONSTANT")
        grad_var_2 = tf.pad(var, [[0, 0], [0, 2], [1, 1], [0, 0]], "CONSTANT")
        grad_var_3 = tf.pad(var, [[0, 0], [1, 1], [2, 0], [0, 0]], "CONSTANT")
        grad_var_4 = tf.pad(var, [[0, 0], [1, 1], [0, 2], [0, 0]], "CONSTANT")
        grad_var = tf.abs(grad_var_nor - grad_var_1) + tf.abs(grad_var_nor - grad_var_2) + \
                   tf.abs(grad_var_nor - grad_var_3) + tf.abs(grad_var_nor - grad_var_4)

        grad_var = tf.gather(grad_var, tf.range(1, tf.shape(grad_var)[1] - 1), axis=1)
        grad_var = tf.gather(grad_var, tf.range(1, tf.shape(grad_var)[2] - 1), axis=2)

        return grad_var

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            yp = K.permute_dimensions(y, r)
            dfi = yp[1:, ...] - yp[:-1, ...]

            if self.vox_weight is not None:
                w = K.permute_dimensions(self.vox_weight, r)
                # TODO: Need to add square root, since for non-0/1 weights this is bad.
                dfi = w[1:, ...] * dfi

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss(self, _, y_pred):
        """
        returns Tensor of size [bs]
        """
        grad_pred = self.gradient(y_pred)
        return tf.reduce_mean(grad_pred * grad_pred)


class SmoothLoss:
    def __init__(self, boundary_awareness=True, alpha=10):
        self.boundary_awareness = boundary_awareness
        self.alpha = alpha
        self.func_smooth = self.smooth_grad_1st

    def smooth_grad_1st(self, flow, image):
        img_dx, img_dy = self.gradient(image)
        dx, dy = self.gradient(flow)
        eps = 1e-6
        dx = tf.sqrt(dx ** 2 + eps)
        dy = tf.sqrt(dy ** 2 + eps)
        dx = tf.abs(dx)
        dy = tf.abs(dy)

        if self.boundary_awareness:
            weights_x = tf.exp(-tf.reduce_mean(tf.abs(img_dx), axis=1, keepdims=True) * self.alpha)
            weights_y = tf.exp(-tf.reduce_mean(tf.abs(img_dy), axis=1, keepdims=True) * self.alpha)
            loss_x = weights_x * dx / 2.
            loss_y = weights_y * dy / 2.
        else:
            loss_x = dx / 2.
            loss_y = dy / 2.

        return tf.reduce_mean(loss_x) / 2. + tf.reduce_mean(loss_y) / 2.

    def loss(self, y_true, y_pred):
        return tf.reduce_mean(self.func_smooth(y_pred, y_true))

    def gradient(self, var):
        grad_var_nor = tf.pad(var, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        grad_var_1 = tf.pad(var, [[0, 0], [2, 0], [1, 1], [0, 0]], "CONSTANT")
        grad_var_2 = tf.pad(var, [[0, 0], [0, 2], [1, 1], [0, 0]], "CONSTANT")
        grad_var_3 = tf.pad(var, [[0, 0], [1, 1], [2, 0], [0, 0]], "CONSTANT")
        grad_var_4 = tf.pad(var, [[0, 0], [1, 1], [0, 2], [0, 0]], "CONSTANT")
        grad_var = tf.abs(grad_var_nor - grad_var_1) + tf.abs(grad_var_nor - grad_var_2) + \
                   tf.abs(grad_var_nor - grad_var_3) + tf.abs(grad_var_nor - grad_var_4)

        grad_var = tf.gather(grad_var, tf.range(1, tf.shape(grad_var)[1] - 1), axis=1)
        grad_var = tf.gather(grad_var, tf.range(1, tf.shape(grad_var)[2] - 1), axis=2)

        return grad_var

class MSE:
    def mse(self, y_true, y_pred):
        return tf.square(y_true - y_pred)

    def loss(self, y_true, y_pred, reduce='mean'):
        # compute mse
        mse = self.mse(y_true, y_pred)
        # reduce
        if reduce == 'mean':
            mse = tf.reduce_mean(mse)
        elif reduce == 'max':
            mse = K.max(mse)
        elif reduce is not None:
            raise ValueError(f'Unknown MSE reduction type: {reduce}')
        # loss
        return mse


class AME:
    def loss(self, _, y_pred, reduce='mean'):
        # compute mse
        ame = tf.abs(y_pred)
        # reduce
        if reduce == 'mean':
            ame = tf.reduce_mean(ame)
        elif reduce == 'max':
            ame = K.max(ame)
        elif reduce is not None:
            raise ValueError(f'Unknown reduction type: {reduce}')
        # loss
        return ame



class MAD:
    def loss(self, _, y_pred):
        # Calculate the absolute difference between each slice along C
        # This will give us a tensor of shape (B, C, C, MM, NN)
        # by broadcasting the original tensor over itself
        diff = tf.expand_dims(y_pred, axis=2) - tf.expand_dims(y_pred, axis=1)

        # Calculate the mean absolute error across the channel dimension
        # This gives us the mean absolute error for each pair of slices along C
        mae = tf.reduce_mean(tf.abs(diff), axis=[1, 3, 4])  # (B, C, C)

        # Average the MAE across all pairs of slices to get a single loss value per batch
        loss = tf.reduce_mean(mae)

        return loss


class PairwiseDiceLoss:
    def pairwise_dice_coefficient(self, y_pred, smooth=1e-5):
        """
        Compute pairwise Dice coefficient across all slices along the channel dimension (C).

        Args:
        y_pred: Tensor of shape (B, C, MM, NN), where B is batch size,
                C is number of channels, and MM, NN are spatial dimensions.

        Returns:
        A single loss value that encourages similarity across all C slices.
        """
        # Expand dimensions to compute pairwise differences along C
        y_pred_1 = tf.expand_dims(y_pred, axis=2)  # Shape: (B, C, 1, MM, NN)
        y_pred_2 = tf.expand_dims(y_pred, axis=1)  # Shape: (B, 1, C, MM, NN)

        # Compute intersection and union for Dice similarity
        intersection = tf.reduce_sum(y_pred_1 * y_pred_2, axis=[3, 4])  # (B, C, C)
        union = tf.reduce_sum(y_pred_1, axis=[3, 4]) + tf.reduce_sum(y_pred_2, axis=[3, 4])  # (B, C, C)

        # Compute pairwise Dice coefficient
        dice = (2. * intersection + smooth) / (union + smooth)  # (B, C, C)

        # Return the mean pairwise Dice coefficient (higher = better alignment)
        return tf.reduce_mean(dice)

    def dice_coefficient_with_reference(self, y_pred, smooth=1e-5):
        """
        Compute Dice coefficient for each channel by comparing it to the mask at C=0.

        Args:
        y_pred: Tensor of shape (B, C, MM, NN), where B is batch size,
                C is number of channels, and MM, NN are spatial dimensions.

        Returns:
        A single loss value that encourages similarity of all C slices to the reference mask at C=0.
        """
        # Extract reference mask (C=0)
        ref_mask = tf.expand_dims(y_pred[:, 0, :, :], axis=1)  # Shape: (B, 1, MM, NN)

        # Compute intersection and union for each channel with the reference
        intersection = tf.reduce_sum(ref_mask * y_pred, axis=[2, 3])  # (B, C)
        union = tf.reduce_sum(ref_mask, axis=[2, 3]) + tf.reduce_sum(y_pred, axis=[2, 3])  # (B, C)

        # Compute Dice coefficient per channel
        dice = (2. * intersection + smooth) / (union + smooth)  # (B, C)

        # Return mean Dice coefficient across all channels (except the reference channel itself)
        return tf.reduce_mean(dice[:, 1:])  # Exclude C=0 from averaging

    def loss(self, _, y_pred):
        """
        Compute the pairwise Dice loss, where minimizing it enforces similarity across C slices.
        """
        return 1 - self.dice_coefficient_with_reference(y_pred)


class dice_loss:
    """
    N-D dice for segmentation
    """
    def dice_coefficient(self, y_true, y_pred, smooth=1):
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice

    def loss(self, y_true, y_pred):
        return 1 - self.dice_coefficient(y_true, y_pred)

class DiceLossMultiContrast:
    """
    N-D Dice loss for segmentation masks with batch and multi-class support
    """
    def dice_coefficient(self, y_true, y_pred, smooth=1e-6):
        # Compute per-class Dice coefficient
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])  # Sum over spatial dimensions MM, NN
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])  # Sum over spatial dimensions

        dice = (2. * intersection + smooth) / (union + smooth)  # Per class
        return tf.reduce_mean(dice, axis=-1)  # Average over classes

    def loss(self, y_true, y_pred):
        dice_coeff = self.dice_coefficient(y_true, y_pred)
        return 1 - tf.reduce_mean(dice_coeff)  # Average over batch


class design_loss():

    def __init__(self, parameter=1, parameter_mi=1, win=9, parameter_threth=0.1):
        self.parameter = parameter
        self.parameter_mi = parameter_mi
        self.win = [win, win]
        self.jl_threth = parameter_threth
        self.MSE = MSE().loss
        self.mi = mi().loss

    def _clip(self, y_true):
        threth = self.jl_threth
        y_round = K.round((K.clip(y_true, 0, threth * 2)) / (threth * 2))
        return y_round

    def mi_clipmse(self, y_true, y_pred):
        if y_true.shape[-1] != y_pred.shape[-1]:
            y_true = tf.repeat(y_true, repeats=y_pred.shape[-1], axis=-1)
        round = self._clip(y_true)
        y_true, y_pred = tf.cast(y_true, tf.float64), tf.cast(y_pred, tf.float64)
        return self.parameter * self.MSE((1 - round) * y_true, (1 - round) * y_pred) + self.parameter_mi * self.mi(y_true, y_pred)


def residuce_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred))

def none_loss(y_true, y_pred):
    return tf.convert_to_tensor([0.0])


def PCA_groupwise(_, y_pred):
    """
    Computes the PCA loss for a batch of predicted and true images.

    Args:
        y_true: Tensor of shape (B, H, W, 1), ground truth images.
        y_pred: Tensor of shape (B, H, W, 1), predicted images.

    Returns:
        L_pca: Tensor of shape (B,), the PCA loss for each batch element.
    """
    # Reshape the inputs to (B, H*W, 1) to prepare for PCA computation
    shape = tf.shape(y_pred)
    #print(shape)
    C, H, W = shape[-4], shape[-3], shape[-2]
    y_pred_t = tf.transpose(y_pred, perm=[0, 4, 2, 3, 1])
    #y_true_flat = tf.reshape(y_true, (-1, H * W))
    M = tf.reshape(y_pred_t, (-1, H * W, C)) #(B, H*W, C)

    # Stack y_true and y_pred along the last axis to form M of shape (B, H*W, 2)
    #M = tf.stack([y_true_flat, y_pred_flat], axis=-1)  # Shape (B, H*W, 2)

    # Step 1: Compute the column-wise mean matrix (Mt) for each batch
    Mt = tf.reduce_mean(M, axis=1, keepdims=True)  # Shape (B, 1, 2)

    # Step 2: Center the data by subtracting the column-wise mean
    M_centered = M - Mt  # Shape (B, H*W, 2)

    # Step 3: Compute the standard deviation for each column in the batch
    sigma = tf.math.reduce_std(M, axis=1, keepdims=True)  # Shape (B, 1, 2)

    # Step 4: Form the diagonal matrix Σ^(-1) for each batch
    Sigma_inv = tf.linalg.diag(1 / sigma[:, 0, :])  # Shape (B, 2, 2)

    # Step 5: Normalize M_centered by Σ^(-1) (matrix multiplication)
    M_normalized = tf.matmul(M_centered, Sigma_inv)  # Shape (B, H*W, 2)

    # Step 6: Compute the normalized correlation matrix K for each batch
    num_samples = tf.cast(H*W - 1, tf.float32)
    K = (1 / num_samples) * tf.matmul(M_normalized, M_normalized, transpose_a=True)  # Shape (B, 2, 2)

    # Step 6a: Symmetrize K to ensure numerical stability
    #K_symmetric = 0.5 * (K + tf.linalg.matrix_transpose(K))  # Symmetrize K

    # Step 7: Perform eigendecomposition on K_symmetric for each batch
    eigenvalues, _ = tf.linalg.eigh(K)  # Shape (B, 2)

    # Step 8: Sort eigenvalues in descending order
    sorted_indices = tf.argsort(eigenvalues, direction='DESCENDING', axis=-1)  # Shape (B, 2)
    sorted_eigenvalues = tf.gather(eigenvalues, sorted_indices, batch_dims=1)  # Shape (B, 2)

    # Step 9: Compute L_pca for the first 2 eigenvalues for each batch
    weights = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)  # Weights for eigenvalues (1-based index)
    L_pca = tf.reduce_sum(sorted_eigenvalues * weights, axis=-1)  # Shape (B,)
    #mean_loss = tf.reduce_mean(L_pca)
    #print(L_pca, mean_loss)
    return L_pca


def PCA_template(y_true, y_pred):
    """
    Computes the PCA loss for a batch of predicted and true images.

    Args:
        y_true: Tensor of shape (B, H, W, 1), ground truth images.
        y_pred: Tensor of shape (B, H, W, 1), predicted images.

    Returns:
        L_pca: Tensor of shape (B,), the PCA loss for each batch element.
    """
    # Reshape the inputs to (B, H*W, 1) to prepare for PCA computation
    shape = tf.shape(y_true)
    #print(shape)
    H, W = shape[-3], shape[-2]

    y_true_flat = tf.reshape(y_true, (-1, H * W))
    y_pred_flat = tf.reshape(y_pred, (-1, H * W))

    # Stack y_true and y_pred along the last axis to form M of shape (B, H*W, 2)
    M = tf.stack([y_true_flat, y_pred_flat], axis=-1)  # Shape (B, H*W, 2)

    # Step 1: Compute the column-wise mean matrix (Mt) for each batch
    Mt = tf.reduce_mean(M, axis=1, keepdims=True)  # Shape (B, 1, 2)

    # Step 2: Center the data by subtracting the column-wise mean
    M_centered = M - Mt  # Shape (B, H*W, 2)

    # Step 3: Compute the standard deviation for each column in the batch
    sigma = tf.math.reduce_std(M, axis=1, keepdims=True)  # Shape (B, 1, 2)

    # Step 4: Form the diagonal matrix Σ^(-1) for each batch
    Sigma_inv = tf.linalg.diag(1 / sigma[:, 0, :])  # Shape (B, 2, 2)

    # Step 5: Normalize M_centered by Σ^(-1) (matrix multiplication)
    M_normalized = tf.matmul(M_centered, Sigma_inv)  # Shape (B, H*W, 2)

    # Step 6: Compute the normalized correlation matrix K for each batch
    num_samples = tf.cast(H*W - 1, tf.float32)
    K = (1 / num_samples) * tf.matmul(M_normalized, M_normalized, transpose_a=True)  # Shape (B, 2, 2)

    # Step 6a: Symmetrize K to ensure numerical stability
    #K_symmetric = 0.5 * (K + tf.linalg.matrix_transpose(K))  # Symmetrize K

    # Step 7: Perform eigendecomposition on K_symmetric for each batch
    eigenvalues, _ = tf.linalg.eigh(K)  # Shape (B, 2)

    # Step 8: Sort eigenvalues in descending order
    sorted_indices = tf.argsort(eigenvalues, direction='DESCENDING', axis=-1)  # Shape (B, 2)
    sorted_eigenvalues = tf.gather(eigenvalues, sorted_indices, batch_dims=1)  # Shape (B, 2)

    # Step 9: Compute L_pca for the first 2 eigenvalues for each batch
    weights = tf.constant([1, 2], dtype=tf.float32)  # Weights for eigenvalues (1-based index)
    L_pca = tf.reduce_sum(sorted_eigenvalues * weights, axis=-1)  # Shape (B,)
    #mean_loss = tf.reduce_mean(L_pca)
    #print(L_pca, mean_loss)
    return L_pca


def PCA_DIXON(y_true, y_pred):
    """
    Computes the PCA loss for a batch of predicted and true images.

    Args:
        y_true: Tensor of shape (B, H, W, 1), ground truth images.
        y_pred: Tensor of shape (B, H, W, 1), predicted images.

    Returns:
        L_pca: Tensor of shape (B,), the PCA loss for each batch element.
    """
    # Reshape the inputs to (B, H*W, 1) to prepare for PCA computation
    shape = tf.shape(y_pred)
    #print(shape)
    C, H, W = shape[-4], shape[-3], shape[-2]
    #print(y_true.shape,  y_pred.shape)
    #y_true = tf.tile(y_true[:,0][:,None], [1, C, 1, 1, 1])
    #print(y_true.shape, y_pred.shape)
    # y_pred = y_pred[:,1:]
    y_true_flat = tf.reshape(y_true, (-1, H * W))
    y_pred_flat = tf.reshape(y_pred, (-1, H * W))

    # Stack y_true and y_pred along the last axis to form M of shape (B, H*W, 2)
    M = tf.stack([y_true_flat, y_pred_flat], axis=-1)  # Shape (B, H*W, 2)

    # Step 1: Compute the column-wise mean matrix (Mt) for each batch
    Mt = tf.reduce_mean(M, axis=1, keepdims=True)  # Shape (B, 1, 2)

    # Step 2: Center the data by subtracting the column-wise mean
    M_centered = M - Mt  # Shape (B, H*W, 2)

    # Step 3: Compute the standard deviation for each column in the batch
    sigma = tf.math.reduce_std(M, axis=1, keepdims=True)  # Shape (B, 1, 2)

    # Step 4: Form the diagonal matrix Σ^(-1) for each batch
    Sigma_inv = tf.linalg.diag(1 / sigma[:, 0, :])  # Shape (B, 2, 2)

    # Step 5: Normalize M_centered by Σ^(-1) (matrix multiplication)
    M_normalized = tf.matmul(M_centered, Sigma_inv)  # Shape (B, H*W, 2)

    # Step 6: Compute the normalized correlation matrix K for each batch
    num_samples = tf.cast(H*W - 1, tf.float32)
    K = (1 / num_samples) * tf.matmul(M_normalized, M_normalized, transpose_a=True)  # Shape (B, 2, 2)

    # Step 6a: Symmetrize K to ensure numerical stability
    #K_symmetric = 0.5 * (K + tf.linalg.matrix_transpose(K))  # Symmetrize K

    # Step 7: Perform eigendecomposition on K_symmetric for each batch
    eigenvalues, _ = tf.linalg.eigh(K)  # Shape (B, 2)

    # Step 8: Sort eigenvalues in descending order
    sorted_indices = tf.argsort(eigenvalues, direction='DESCENDING', axis=-1)  # Shape (B, 2)
    sorted_eigenvalues = tf.gather(eigenvalues, sorted_indices, batch_dims=1)  # Shape (B, 2)

    # Step 9: Compute L_pca for the first 2 eigenvalues for each batch
    weights = tf.constant([1, 2], dtype=tf.float32)  # Weights for eigenvalues (1-based index)
    L_pca = tf.reduce_sum(sorted_eigenvalues * weights, axis=-1)  # Shape (B,)
    #mean_loss = tf.reduce_mean(L_pca)
    #print(L_pca, mean_loss)
    return L_pca



def PCA_to_DIXON(_, y_pred):
    """
    Computes the PCA loss for a batch of predicted and true images.

    Args:
        y_true: Tensor of shape (B, H, W, 1), ground truth images.
        y_pred: Tensor of shape (B, H, W, 1), predicted images.

    Returns:
        L_pca: Tensor of shape (B,), the PCA loss for each batch element.
    """
    # Reshape the inputs to (B, H*W, 1) to prepare for PCA computation
    shape = tf.shape(y_pred)
    #print(shape)
    C, H, W = shape[-1], shape[-3], shape[-2]
    #print(y_true.shape,  y_pred.shape)
    y_true = tf.tile(y_pred[...,0][...,None], [1, 1, 1, C])
    #print(y_true.shape, y_pred.shape)
    # y_pred = y_pred[:,1:]
    y_true = tf.transpose(y_true, (0,3,1,2))
    y_pred = tf.transpose(y_pred, (0,3,1,2))
    y_true_flat = tf.reshape(y_true, (-1, H * W))
    y_pred_flat = tf.reshape(y_pred, (-1, H * W))

    # Stack y_true and y_pred along the last axis to form M of shape (B, H*W, 2)
    M = tf.stack([y_true_flat, y_pred_flat], axis=-1)  # Shape (B, H*W, 2)

    # Step 1: Compute the column-wise mean matrix (Mt) for each batch
    Mt = tf.reduce_mean(M, axis=1, keepdims=True)  # Shape (B, 1, 2)

    # Step 2: Center the data by subtracting the column-wise mean
    M_centered = M - Mt  # Shape (B, H*W, 2)

    # Step 3: Compute the standard deviation for each column in the batch
    sigma = tf.math.reduce_std(M, axis=1, keepdims=True)  # Shape (B, 1, 2)

    # Step 4: Form the diagonal matrix Σ^(-1) for each batch
    Sigma_inv = tf.linalg.diag(1 / sigma[:, 0, :])  # Shape (B, 2, 2)

    # Step 5: Normalize M_centered by Σ^(-1) (matrix multiplication)
    M_normalized = tf.matmul(M_centered, Sigma_inv)  # Shape (B, H*W, 2)

    # Step 6: Compute the normalized correlation matrix K for each batch
    num_samples = tf.cast(H*W - 1, tf.float32)
    K = (1 / num_samples) * tf.matmul(M_normalized, M_normalized, transpose_a=True)  # Shape (B, 2, 2)

    # Step 6a: Symmetrize K to ensure numerical stability
    #K_symmetric = 0.5 * (K + tf.linalg.matrix_transpose(K))  # Symmetrize K

    # Step 7: Perform eigendecomposition on K_symmetric for each batch
    eigenvalues, _ = tf.linalg.eigh(K)  # Shape (B, 2)

    # Step 8: Sort eigenvalues in descending order
    sorted_indices = tf.argsort(eigenvalues, direction='DESCENDING', axis=-1)  # Shape (B, 2)
    sorted_eigenvalues = tf.gather(eigenvalues, sorted_indices, batch_dims=1)  # Shape (B, 2)

    # Step 9: Compute L_pca for the first 2 eigenvalues for each batch
    weights = tf.constant([1, 2], dtype=tf.float32)  # Weights for eigenvalues (1-based index)
    L_pca = tf.reduce_sum(sorted_eigenvalues * weights, axis=-1)  # Shape (B,)
    #mean_loss = tf.reduce_mean(L_pca)
    #print(L_pca, mean_loss)
    return L_pca



class MutualInformationVXM(ne.metrics.MutualInformation):
    """
    Soft Mutual Information approximation for intensity volumes

    More information/citation:
    - Courtney K Guo.
      Multi-modal image registration with unsupervised deep learning.
      PhD thesis, Massachusetts Institute of Technology, 2019.
    - M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
      SynthMorph: learning contrast-invariant registration without acquired images
      IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
      https://doi.org/10.1109/TMI.2021.3116879
    """

    def loss(self, y_true, y_pred):
        #print(y_pred.shape, y_true.shape)
        return -self.volumes(y_true, y_pred)


class MI_PCA_loss:
    def loss(self,y_true, y_pred):
        L_pca = PCA_DIXON(y_true, y_pred)
        mean_loss = tf.reduce_mean(L_pca)
        return MutualInformationVXM().loss(y_true, y_pred) + mean_loss
