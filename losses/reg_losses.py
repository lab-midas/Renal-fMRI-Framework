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




class MutualInformation:
    def __init__(self, num_bins=100, sigma_ratio=1):
        super(MutualInformation, self).__init__()
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









class DiceLoss2Classes:
    """
    Computes Dice loss for multi-class segmentation (excluding background).
    Assumes input values are in {0, 1, 2}.
    """

    def dice_coefficient(self, y_true, y_pred, smooth=1e-6):
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice  # Shape: (B, num_classes)

    def loss(self, y_true, y_pred):
        # Convert integer labels to one-hot encoding (shape: B, MM, NN, num_classes)
        y_true_one_hot = tf.one_hot(tf.cast(y_true[..., 0], tf.int32), depth=3)
        y_pred_one_hot = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=3)

        # Compute dice per class
        dice_per_class = self.dice_coefficient(y_true_one_hot, y_pred_one_hot)

        # Ignore background class (index 0), compute mean over class 1 and 2
        dice_loss = 1 - tf.reduce_mean(dice_per_class[:, 1:], axis=-1)  # Shape: (B,)

        return tf.reduce_mean(dice_loss)  # Scalar loss


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




def residuce_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred))

def none_loss(y_true, y_pred):
    return tf.convert_to_tensor([0.0])



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






class MutualInformation2(ne.metrics.MutualInformation):
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

