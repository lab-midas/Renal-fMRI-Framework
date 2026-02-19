import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

alpha = 0.25
gamma = 2

class lossObj:
    def __init__(self):
        self.alpha = 0.5
        print('Loss init.')
        class_weights = tf.constant([0.01, 1.0, 1.0, 1.0, 1.0])
        self.class_weights = class_weights / tf.reduce_sum(class_weights)
        print(self.class_weights)

        # Create an image of `sample_weights` by using the label at each pixel as an
        # index into the `class weights` .


    def tversky_loss(self, y_true, y_pred):
        '''
        # Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
        # -> the score is computed for each class separately and then summed
        # alpha=beta=0.5 : dice coefficient
        # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
        # alpha+beta=1   : produces set of F*-scores
        # using alpha = 0.3, beta=0.7 from recommendation in https://arxiv.org/pdf/1706.05721.pdf
        '''
        alpha = 0.3
        beta = 0.7

        ones = tf.cast(tf.ones_like(y_true), tf.float32)
        p0 = y_pred  # prob. that voxels are class i
        p1 = ones - y_pred  # prob. that voxels are not class i
        g0 = y_true
        g1 = ones - y_true

        num = tf.cast(tf.reduce_sum(tf.multiply(p0, g0), axis=[0, 1, 2]), tf.float32)
        term1 = tf.cast(alpha * tf.reduce_sum(tf.multiply(p0, g1), axis=[0, 1, 2]), tf.float32)
        term2 = tf.cast(beta * tf.reduce_sum(tf.multiply(p1, g0), axis=[0, 1, 2]), tf.float32)
        den = num + term1 + term2

        loss_term0 = tf.reduce_sum(num[0] / den[0])
        loss_term1 = tf.reduce_sum(num[1] / den[1])
        loss_term2 = tf.reduce_sum(num[2] / den[2])
        # when summing over classes, T has dynamic range [0 Ncl]
        loss_term = self.class_weights[0]*loss_term0 + self.class_weights[1]*loss_term1 + self.class_weights[2]*loss_term2
        #loss_term = tf.reduce_sum(num / den)
        tv_loss = 1 - loss_term
        #cfl = SparseCategoricalFocalLoss(from_logits=True, gamma=2.0)
        cce = SparseCategoricalCrossentropy(from_logits=True)
        y_true = tf.math.argmax(y_true, axis=-1)[..., None]

        bce_loss = cce(y_true, y_pred)
        #cfl_loss = cfl(y_true, y_pred)
        return tv_loss + bce_loss #+ 0.5 * cfl_loss


    def tversky_loss_lbl5(self, y_true, y_pred):
        alpha = 0.3
        beta = 0.7

        ones = tf.cast(tf.ones_like(y_true), tf.float32)
        p0 = y_pred  # prob. that voxels are class i
        p1 = ones - y_pred  # prob. that voxels are not class i
        g0 = y_true
        g1 = ones - y_true

        num = tf.cast(tf.reduce_sum(tf.multiply(p0, g0), axis=[0, 1, 2]), tf.float32)
        term1 = tf.cast(alpha * tf.reduce_sum(tf.multiply(p0, g1), axis=[0, 1, 2]), tf.float32)
        term2 = tf.cast(beta * tf.reduce_sum(tf.multiply(p1, g0), axis=[0, 1, 2]), tf.float32)
        den = num + term1 + term2

        loss_term0 = tf.reduce_sum(num[0] / den[0])
        loss_term1 = tf.reduce_sum(num[1] / den[1])
        loss_term2 = tf.reduce_sum(num[2] / den[2])
        loss_term3 = tf.reduce_sum(num[3] / den[3])
        loss_term4 = tf.reduce_sum(num[4] / den[4])
        # when summing over classes, T has dynamic range [0 Ncl]
        loss_term = (self.class_weights[0]*loss_term0 +
                     self.class_weights[1]*loss_term1 +
                     self.class_weights[2]*loss_term2 +
                     self.class_weights[3] * loss_term3 +
                     self.class_weights[4] * loss_term4)
        #loss_term = tf.reduce_sum(num / den)
        tv_loss = 1 - loss_term
        #cfl = SparseCategoricalFocalLoss(from_logits=True, gamma=2.0)
        cce = SparseCategoricalCrossentropy(from_logits=True)
        y_true = tf.math.argmax(y_true, axis=-1)[..., None]

        bce_loss = cce(y_true, y_pred)
        #cfl_loss = cfl(y_true, y_pred)
        return tv_loss + bce_loss #+ 0.5 * cfl_loss



    
 