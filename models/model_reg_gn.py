import warnings

import neurite as ne
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate, Activation
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import MaxPooling2D, Lambda
from tensorflow_addons.layers import GroupNormalization

###############################################################################
# Registration network
###############################################################################


class modelObj:
    def __init__(self, cfg, kernel_init=None):
        self.img_size_x   = cfg.img_size_x
        self.img_size_y   = cfg.img_size_y
        self.num_channels = cfg.num_channels
        self.conv_kernel  = (3,3)
        self.num_classes = cfg.num_classes
        self.num_contrasts = cfg.num_contrasts
        self.no_filters   = [1, 16, 32, 64, 128, 128]
        self.thres_coef = tf.Variable(tf.constant(tf.random.normal([1]), shape=(1,), dtype=tf.float32), trainable=True)
        if kernel_init == None:
            self.kernel_init = tf.keras.initializers.HeNormal(seed=1)
        else:
            self.kernel_init = kernel_init
        
    def encoder_block_contract(self, block_input, num_fts, pool_flag=True, block_name=1):        
        ''' Defining a UNET block in the feature downsampling path '''
        conv_kernel = self.conv_kernel
        num_groups = num_fts // 4 if num_fts >= 4 else num_fts
        if pool_flag:
            block_input = MaxPooling2D(pool_size=(2,2), name='enc_mp_' + str(block_name))(block_input)
            
        down = Conv2D(num_fts, conv_kernel, padding='same', name = 'enc_conv1_'+ str(block_name), kernel_initializer=self.kernel_init)(block_input)
        down = GroupNormalization(groups = num_groups, name = 'enc_gn1_'+ str(block_name))(down)
        down = Activation('relu',name = 'enc_act1_'+ str(block_name))(down)
        down = Conv2D(num_fts, conv_kernel, padding='same', name = 'enc_conv2_'+ str(block_name), kernel_initializer=self.kernel_init)(down)
        down = GroupNormalization(groups = num_groups, name = 'enc_gn2_'+ str(block_name))(down)
        down = Activation('relu', name = 'enc_act2_'+ str(block_name ))(down)
        return down

    def decoder_block_expand(self, block_input, numFts, concat_block, upsample_flag=True, block_name=1):
        # Defining a UNET block in the feature upsampling path
        conv_kernel = self.conv_kernel
        num_groups = numFts // 4
        if upsample_flag:
            block_input = UpSampling2D(size=(2,2),name='dec_upsamp_'+ str(block_name))(block_input)
            block_input = Conv2D(numFts, kernel_size=(2,2),padding='same', name = 'dec_upsamp_conv_'+ str(block_name), kernel_initializer=self.kernel_init)(block_input)
            block_input = GroupNormalization(groups = num_groups, name = 'dec_upsamp_gn_'+ str(block_name))(block_input)
            block_input = Activation('relu', name = 'dec_upsamp_act_'+ str(block_name ))(block_input)
            block_input = Concatenate(axis=-1, name = 'dec_concat_'+ str(block_name))([block_input, concat_block])
        up = Conv2D(numFts,conv_kernel,padding='same', name = 'dec_conv1_'+ str(block_name), kernel_initializer=self.kernel_init)(block_input)
        up = GroupNormalization(groups = num_groups, name = 'dec_gn1_'+ str(block_name))(up)
        up = Activation('relu', name = 'dec_act1_'+ str(block_name ))(up)
        up = Conv2D(numFts,conv_kernel,padding='same', name = 'dec_conv2_'+ str(block_name), kernel_initializer=self.kernel_init)(up)
        up = GroupNormalization(groups = num_groups, name = 'dec_gn2_'+ str(block_name))(up)
        up = Activation('relu', name = 'dec_act2_'+ str(block_name))(up)
        return up



    def reg_unet(self, weighted=False):
        no_filters = self.no_filters
        im_fix = Input((self.img_size_x, self.img_size_y, self.num_channels))  # fixed
        im_mov = Input((self.img_size_x, self.img_size_y, self.num_channels))  # moving
        lbl_mov = Input((self.img_size_x, self.img_size_y, self.num_channels))
        if weighted:
            weight = Input((self.img_size_x, self.img_size_y, self.num_channels))
        inputs = tf.concat([im_fix, im_mov], axis=3)

        # Encoder fine network
        enc_c1 = self.encoder_block_contract(inputs, no_filters[1], pool_flag=False, block_name=1)
        enc_c2 = self.encoder_block_contract(enc_c1, no_filters[2], block_name=2)
        enc_c3 = self.encoder_block_contract(enc_c2, no_filters[3], block_name=3)
        enc_c4 = self.encoder_block_contract(enc_c3, no_filters[4], block_name=4)
        enc_c5 = self.encoder_block_contract(enc_c4, no_filters[5], block_name=5)
        enc_c6 = self.encoder_block_contract(enc_c5, no_filters[5], block_name=6)

        ###################################
        # Decoder network - Upsampling Path
        ###################################
        dec_c5 = self.decoder_block_expand(enc_c6, no_filters[5], enc_c5, block_name=5)        
        dec_c4 = self.decoder_block_expand(dec_c5, no_filters[4], enc_c4, block_name=4)
        dec_c3 = self.decoder_block_expand(dec_c4, no_filters[3], enc_c3, block_name=3)
        dec_c2 = self.decoder_block_expand(dec_c3, no_filters[2], enc_c2, block_name=2)
        dec_c1 = self.decoder_block_expand(dec_c2, no_filters[1], enc_c1, block_name=1)

        model_op = Conv2D(16, 3, padding='same', kernel_initializer=self.kernel_init)(dec_c1)
        reg_op2 = Activation('relu')(model_op)
        flo_forward = Conv2D(2, 1, name='reg_layer', padding='same', use_bias=False, kernel_initializer=self.kernel_init)(reg_op2)
        flo_backward = flowinverse(flo_forward)

        warped_im_mov = SpatialTransformer(fill_value=0, shape=(self.img_size_x, self.img_size_y), shift_center=False, name='stn_mov_im')((im_mov, flo_forward))


        warped_lbl_mov = SpatialTransformer(interp_method='nearest', fill_value=0, shape=(self.img_size_x, self.img_size_y), shift_center=False, name='stn_mov_lbl')((lbl_mov, flo_forward))
        inverse_warped_moving = SpatialTransformer(fill_value=0, shape=(self.img_size_x, self.img_size_y), shift_center=False, name='defo_iv')([warped_im_mov, flo_backward])
        resduce = Lambda(lambda x: x[0] - x[1], name='rl')([im_mov, inverse_warped_moving])

        if weighted:
            warped_im_mov = Lambda(lambda x: x[0] * x[1], name='warped_im_mov')([weight, warped_im_mov])
            resduce = Lambda(lambda x: x[0] * x[1], name='resduce')([weight, resduce])
            flo_forward = Lambda(lambda x: x[0] * x[1], name='flo_forward')([weight, flo_forward])
            model = Model(inputs=[im_fix, im_mov, lbl_mov, weight], outputs=[warped_im_mov, warped_lbl_mov, flo_forward, resduce]) # pred_fix_affine
        else:
            model = Model(inputs=[im_fix, im_mov, lbl_mov], outputs=[warped_im_mov, warped_lbl_mov, flo_forward, resduce])
        return model

    def reg_groupwise(self, test_mode=False, weighted=False, clip=False):
        ## PCA training
        no_filters = self.no_filters
        template = Input((self.num_contrasts, self.img_size_x, self.img_size_y, self.num_channels), name='template')  # moving
        moving = Input((self.num_contrasts, self.img_size_x, self.img_size_y, self.num_channels), name='moving')  # moving
        moving_lbl = Input((self.num_contrasts, self.img_size_x, self.img_size_y, self.num_classes), name='moving_lbl')
        if weighted:
            weights = Input((self.num_contrasts, self.img_size_x, self.img_size_y, self.num_channels), name='weights')

        template_reshaped = Lambda(lambda x: tf.reshape(x, [-1, self.img_size_x, self.img_size_y, self.num_channels]))(template)
        moving_reshaped = Lambda(lambda x: tf.reshape(x, [-1, self.img_size_x, self.img_size_y, self.num_channels]))(moving)
        moving_lbl_reshaped = Lambda(lambda x: tf.reshape(x, [-1, self.img_size_x, self.img_size_y, self.num_classes]))(moving_lbl)
        #print('moving_reshaped', moving_reshaped.shape)
        inputs = tf.concat([template_reshaped, moving_reshaped], axis=-1)
        #print('inputs', inputs.shape)
        # Encoder fine network
        enc_c1 = self.encoder_block_contract(inputs, no_filters[1], pool_flag=False, block_name=1)
        enc_c2 = self.encoder_block_contract(enc_c1, no_filters[2], block_name=2)
        enc_c3 = self.encoder_block_contract(enc_c2, no_filters[3], block_name=3)
        enc_c4 = self.encoder_block_contract(enc_c3, no_filters[4], block_name=4)
        enc_c5 = self.encoder_block_contract(enc_c4, no_filters[5], block_name=5)
        enc_c6 = self.encoder_block_contract(enc_c5, no_filters[5], block_name=6)

        ###################################
        # Decoder network - Upsampling Path
        ###################################
        dec_c5 = self.decoder_block_expand(enc_c6, no_filters[5], enc_c5, block_name=5)
        dec_c4 = self.decoder_block_expand(dec_c5, no_filters[4], enc_c4, block_name=4)
        dec_c3 = self.decoder_block_expand(dec_c4, no_filters[3], enc_c3, block_name=3)
        dec_c2 = self.decoder_block_expand(dec_c3, no_filters[2], enc_c2, block_name=2)
        dec_c1 = self.decoder_block_expand(dec_c2, no_filters[1], enc_c1, block_name=1)

        model_op = Conv2D(16, 3, padding='same', kernel_initializer=self.kernel_init)(dec_c1)
        reg_op2 = Activation('relu')(model_op)
        flo_forward = Conv2D(2, 1, name='pred_flow',padding='same', use_bias=False, kernel_initializer=self.kernel_init)(reg_op2)
        flo_backward = flowinverse(flo_forward)
        #print('flo_forward', flo_forward.shape)
        warped_im_mov_deformable = SpatialTransformer(fill_value=0, shape=(self.img_size_x, self.img_size_y),shift_center=False, name='pred_fix_deformable_batch')((moving_reshaped, flo_forward))
        #print('warped_im_mov_deformable', warped_im_mov_deformable.shape)
        warped_lbl_mov = SpatialTransformer(interp_method='nearest', fill_value=0, shape=(self.img_size_x, self.img_size_y),shift_center=False, name='pred_fix_lbl_group')((moving_lbl_reshaped, flo_forward))
        #print('warped_lbl_mov', warped_lbl_mov.shape)
        inverse_warped_moving = SpatialTransformer(fill_value=0, shape=(self.img_size_x, self.img_size_y), shift_center=False, name='pred_mov_diffeo')([warped_im_mov_deformable, flo_backward])
        #print('inverse_warped_moving', inverse_warped_moving.shape)
        resduce = Lambda(lambda x: x[0] - x[1], name='diffeo')([moving_reshaped, inverse_warped_moving])
        if test_mode:
            flo_forward = Lambda(lambda x: tf.reshape(x, (-1, self.num_contrasts, self.img_size_x, self.img_size_y, 2)), name='pred_flow_reshaped')(flo_forward)

        warped_im_mov_deformable = Lambda(lambda x: tf.reshape(x, (-1, self.num_contrasts, self.img_size_x, self.img_size_y, self.num_channels)), name='pred_fix_deformable')(warped_im_mov_deformable)
        warped_lbl_mov = Lambda(lambda x: tf.reshape(x, (-1, self.num_contrasts, self.img_size_x, self.img_size_y, self.num_classes)), name='pred_fix_lbl')(warped_lbl_mov)

        if weighted:
            warped_im_mov_deformable = Lambda(lambda x: x[0]*x[1], name='weighted_deformable')([warped_im_mov_deformable, weights])
            weights_reshaped = Lambda(lambda x: tf.reshape(x, [-1, self.img_size_x, self.img_size_y, self.num_channels]), name='weights_reshaped')([weights])
            resduce = Lambda(lambda x: x[0]*x[1], name='weighted_res')([resduce, weights_reshaped])
            if not test_mode:
                flo_forward = Lambda(lambda x: x[0]*x[1], name='weighted_flo')([flo_forward, weights_reshaped])
            model = Model(inputs=[template, moving, moving_lbl, weights],
                          outputs=[warped_im_mov_deformable, warped_lbl_mov, flo_forward, resduce])
        else:

            model = Model(inputs=[template, moving, moving_lbl],
                      outputs=[warped_im_mov_deformable, warped_lbl_mov, flo_forward, resduce]) # pred_fix_affine


        return model


    def reg_groupwise_affine(self,weighted=False):
        ## PCA training
        no_filters = self.no_filters
        template = Input((self.num_contrasts, self.img_size_x, self.img_size_y, self.num_channels), name='template')  # moving
        moving = Input((self.num_contrasts, self.img_size_x, self.img_size_y, self.num_channels), name='moving')  # moving
        moving_lbl = Input((self.num_contrasts, self.img_size_x, self.img_size_y, self.num_classes), name='moving_lbl')
        if weighted:
            weights = Input((self.num_contrasts, self.img_size_x, self.img_size_y, self.num_channels), name='weights')

        template_reshaped = Lambda(lambda x: tf.reshape(x, [-1, self.img_size_x, self.img_size_y, self.num_channels]))(template)
        moving_reshaped = Lambda(lambda x: tf.reshape(x, [-1, self.img_size_x, self.img_size_y, self.num_channels]))(moving)
        moving_lbl_reshaped = Lambda(lambda x: tf.reshape(x, [-1, self.img_size_x, self.img_size_y, self.num_classes]))(moving_lbl)
        #print('moving_reshaped', moving_reshaped.shape)
        inputs = tf.concat([template_reshaped, moving_reshaped], axis=-1)
        #print('inputs', inputs.shape)
        # Encoder fine network
        enc_c11 = self.encoder_block_contract(inputs, no_filters[1], pool_flag=False, block_name=11)
        enc_c12 = self.encoder_block_contract(enc_c11, no_filters[2], block_name=12)
        enc_c13 = self.encoder_block_contract(enc_c12, no_filters[3], block_name=13)
        enc_c14 = self.encoder_block_contract(enc_c13, no_filters[4], block_name=14)
        enc_c15 = self.encoder_block_contract(enc_c14, no_filters[5], block_name=15)
        enc_c16 = self.encoder_block_contract(enc_c15, no_filters[5], block_name=16)

        reg_coarse_layer = Conv2D(32, 1, name='reg_coarse_layer', padding='same',
                                  use_bias=True,
                                  kernel_initializer=self.kernel_init)(enc_c16)
        reg_coarse_layer = MaxPooling2D((2, 2))(reg_coarse_layer)
        reg_coarse_layer = tf.keras.layers.Flatten()(reg_coarse_layer)
        reg_coarse_layer = tf.keras.layers.Dense(16, name="fc_1", activation=Activation('relu'))(reg_coarse_layer)
        reg_coarse_layer = tf.keras.layers.Dense(6, name="fc_2")(reg_coarse_layer)
        reg_coarse_layer = tf.keras.layers.Reshape((2, 3))(reg_coarse_layer)
        reg_coarse_layer = AffineToDenseShift((self.img_size_x, self.img_size_y), shift_center=False)(reg_coarse_layer)
        warped_im_mov_affine = SpatialTransformer(fill_value=0, shape=(self.img_size_x, self.img_size_y),
                                                  shift_center=False,
                                                  name='pred_fix_affine')((moving_reshaped, reg_coarse_layer))

        warped_lbl_mov_affine = SpatialTransformer(fill_value=0, shape=(self.img_size_x, self.img_size_y),
                                                  shift_center=False,
                                                  name='pred_lbl_affine')((moving_lbl_reshaped, reg_coarse_layer))

        # warped_im_mov_affine = Lambda(lambda x: tf.reshape(x, (-1, self.num_contrasts, self.img_size_x, self.img_size_y, self.num_channels)), name='pred_fix_deformable')(warped_im_mov_affine)
        # warped_lbl_mov_affine = Lambda(lambda x: tf.reshape(x, (-1, self.num_contrasts, self.img_size_x, self.img_size_y, self.num_classes)), name='pred_fix_lbl')(warped_lbl_mov_affine)

        if weighted:
            weights_reshaped = Lambda(
                lambda x: tf.reshape(x, [-1, self.img_size_x, self.img_size_y, self.num_channels]),
                name='weights_reshaped')([weights])
            warped_im_mov_affine = Lambda(lambda x: x[0] * x[1], name='weighted_deformable')(
                [warped_im_mov_affine, weights_reshaped])

            model = Model(inputs=[template, moving, moving_lbl, weights],
                      outputs=[warped_im_mov_affine, warped_lbl_mov_affine, reg_coarse_layer])
        else:
            model = Model(inputs=[template, moving, moving_lbl],
                          outputs=[warped_im_mov_affine, warped_lbl_mov_affine, reg_coarse_layer])

        return model

    def reg_groupwise_affine_nonrigid(self, checkpoint_affine, test_mode=False, weighted=False):
        ## PCA training
        no_filters = self.no_filters
        template = Input((self.num_contrasts, self.img_size_x, self.img_size_y, self.num_channels),
                         name='template_deformable')  # moving
        moving = Input((self.num_contrasts, self.img_size_x, self.img_size_y, self.num_channels),
                       name='moving_deformable')  # moving
        moving_lbl = Input((self.num_contrasts, self.img_size_x, self.img_size_y, self.num_classes), name='moving_lbl_deformable')
        if weighted:
            weights = Input((self.num_contrasts, self.img_size_x, self.img_size_y, self.num_channels), name='weights_deformable')
            inputs = [template, moving, moving_lbl, weights]
        else:
            inputs = [template, moving, moving_lbl]

        affine_model = self.reg_groupwise_affine(weighted=weighted)
        affine_model.load_weights(checkpoint_affine)

        for layer in affine_model.layers:
            layer.trainable = False

        warped_im_mov_affine, warped_lbl_mov_affine, reg_coarse_layer = affine_model(inputs)
        template_reshaped = Lambda(lambda x: tf.reshape(x, [-1, self.img_size_x, self.img_size_y, self.num_channels]))(template)
        moving_reshaped = Lambda(lambda x: tf.reshape(x, [-1, self.img_size_x, self.img_size_y, self.num_channels]))( moving)
        inputs_deformable = tf.concat([template_reshaped, moving_reshaped], axis=-1)

        # Encoder fine network
        enc_c1 = self.encoder_block_contract(inputs_deformable, no_filters[1], pool_flag=False, block_name=1)
        enc_c2 = self.encoder_block_contract(enc_c1, no_filters[2], block_name=2)
        enc_c3 = self.encoder_block_contract(enc_c2, no_filters[3], block_name=3)
        enc_c4 = self.encoder_block_contract(enc_c3, no_filters[4], block_name=4)
        enc_c5 = self.encoder_block_contract(enc_c4, no_filters[5], block_name=5)
        enc_c6 = self.encoder_block_contract(enc_c5, no_filters[5], block_name=6)
        #print(affine_model.pop())
        # Define the layers to extract
        # layer_names = [
        #     "enc_act2_16",
        #     "enc_act2_15",
        #     "enc_act2_14",
        #     "enc_act2_13",
        #     "enc_act2_12",
        #     "enc_act2_11"
        # ]
        #
        # # Get the layer outputs
        # layer_outputs = [affine_model.get_layer(name=name).output for name in layer_names]
        #
        # # Create a new model with the same inputs but multiple outputs
        # extractor_model = Model(inputs=affine_model.input, outputs=layer_outputs)
        #
        # # Run the model on input data
        # enc_c6, enc_c5, enc_c4, enc_c3, enc_c2, enc_c1 = extractor_model(inputs)

        ###################################
        # Decoder network - Upsampling Path
        ###################################
        dec_c5 = self.decoder_block_expand(enc_c6, no_filters[5], enc_c5, block_name=5)
        dec_c4 = self.decoder_block_expand(dec_c5, no_filters[4], enc_c4, block_name=4)
        dec_c3 = self.decoder_block_expand(dec_c4, no_filters[3], enc_c3, block_name=3)
        dec_c2 = self.decoder_block_expand(dec_c3, no_filters[2], enc_c2, block_name=2)
        dec_c1 = self.decoder_block_expand(dec_c2, no_filters[1], enc_c1, block_name=1)

        model_op = Conv2D(16, 3, padding='same', kernel_initializer=self.kernel_init)(dec_c1)
        reg_op2 = Activation('relu')(model_op)
        flo_forward = Conv2D(2, 1, name='pred_flow', padding='same', use_bias=False,
                             kernel_initializer=self.kernel_init)(reg_op2)
        #flo_forward = Lambda(lambda x: x[0] + x[1])([flo_forward, reg_coarse_layer])

        flo_backward = flowinverse(flo_forward)
        # print('flo_forward', flo_forward.shape)
        warped_im_mov_deformable = SpatialTransformer(fill_value=0, shape=(self.img_size_x, self.img_size_y),
                                                      shift_center=False, name='pred_fix_deformable_batch')(
            (warped_im_mov_affine, flo_forward))
        # print('warped_im_mov_deformable', warped_im_mov_deformable.shape)
        warped_lbl_mov = SpatialTransformer(interp_method='nearest', fill_value=0,
                                            shape=(self.img_size_x, self.img_size_y), shift_center=False,
                                            name='pred_fix_lbl_group')((warped_lbl_mov_affine, flo_forward))
        # print('warped_lbl_mov', warped_lbl_mov.shape)
        inverse_warped_moving = SpatialTransformer(fill_value=0, shape=(self.img_size_x, self.img_size_y),
                                                   shift_center=False, name='pred_mov_diffeo')(
            [warped_im_mov_deformable, flo_backward])
        # print('inverse_warped_moving', inverse_warped_moving.shape)
        resduce = Lambda(lambda x: x[0] - x[1], name='diffeo')([warped_im_mov_affine, inverse_warped_moving])

        if test_mode:
            flo_forward = Lambda(lambda x: tf.reshape(x, (-1, self.num_contrasts, self.img_size_x, self.img_size_y, 2)),
                                 name='pred_flow_reshaped')(flo_forward)

        warped_im_mov_deformable = Lambda(
            lambda x: tf.reshape(x, (-1, self.num_contrasts, self.img_size_x, self.img_size_y, self.num_channels)),
            name='pred_fix_deformable')(warped_im_mov_deformable)
        warped_lbl_mov = Lambda(
            lambda x: tf.reshape(x, (-1, self.num_contrasts, self.img_size_x, self.img_size_y, self.num_classes)),
            name='pred_fix_lbl')(warped_lbl_mov)



        if weighted and not test_mode:
            warped_im_mov_deformable = Lambda(lambda x: x[0] * x[1], name='weighted_deformable')(
                [warped_im_mov_deformable, weights])
            weights_reshaped = Lambda(
                lambda x: tf.reshape(x, [-1, self.img_size_x, self.img_size_y, self.num_channels]),
                name='weights_reshaped')([weights])
            resduce = Lambda(lambda x: x[0] * x[1], name='weighted_res')([resduce, weights_reshaped])
            flo_forward = Lambda(lambda x: x[0] * x[1], name='weighted_flo')([flo_forward, weights_reshaped])

        model = Model(inputs=inputs,
                      outputs=[warped_im_mov_deformable, warped_lbl_mov, flo_forward, resduce])  # pred_fix_affine

        return model

###############################################################################
# Spatial Transformer
###############################################################################


class SpatialTransformer(Layer):
    """
    N-dimensional (ND) spatial transformer layer

    Applies affine and dense transforms to images. A dense transform gives
    displacements (not absolute locations) at each voxel.

    If you find this layer useful, please cite:

      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    Originally, this code was based on voxelmorph code, which
    was in turn transformed to be dense with the help of (affine) STN code
    via https://github.com/kevinzakka/spatial-transformer-network.

    Since then, we've re-written the code to be generalized to any
    dimensions, and along the way wrote grid and interpolation functions.
    """

    def __init__(self,
                 interp_method='linear',
                 single_transform=False,
                 fill_value=None,
                 shift_center=True,
                 shape=None,
                 **kwargs):
        """
        Parameters:
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            single_transform: Use single transform for the entire image batch.
            fill_value: Value to use for points sampled outside the domain.
                If None, the nearest neighbors will be used.
            shift_center: Shift grid to image center when converting affine
                transforms to dense transforms. Assumes the input and output spaces are identical.
            shape: ND output shape used when converting affine transforms to dense
                transforms. Includes only the N spatial dimensions. If None, the
                shape of the input image will be used. Incompatible with `shift_center=True`.

        Notes:
            There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
            indexing. Due to inconsistencies in how some functions and layers handled xy-indexing,
            we removed it in favor of default ij-indexing to minimize the potential for confusion.

        """
        # TODO: remove this block
        # load models saved with the `indexing` argument
        if 'indexing' in kwargs:
            del kwargs['indexing']
            warnings.warn('The `indexing` argument to SpatialTransformer no longer exists. If you '
                          'loaded a model, save it again to be able to load it in the future.')

        self.interp_method = interp_method
        self.single_transform = single_transform
        self.fill_value = fill_value
        self.shift_center = shift_center
        self.shape = shape
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'single_transform': self.single_transform,
            'fill_value': self.fill_value,
            'shift_center': self.shift_center,
            'shape': self.shape,
        })
        return config

    def build(self, input_shape):

        # sanity check on input list
        if len(input_shape) > 2:
            raise ValueError('Spatial Transformer must be called on a list of length 2: '
                             'first argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.imshape = input_shape[0][1:]
        self.trfshape = input_shape[1][1:]

        # make sure transform has reasonable shape (is_affine_shape throws error if not)
        if not is_affine_shape(input_shape[1][1:]):
            image_shape = tuple(self.imshape[:-1])
            dense_shape = tuple(self.trfshape[:-1])
            if image_shape != dense_shape:
                warnings.warn(f'Dense transform shape {dense_shape} does not match '
                              f'image shape {image_shape}.')

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: List of [img, trf], where img is the ND moving image and trf
            is either a dense warp of shape [B, D1, ..., DN, N] or an affine matrix
            of shape [B, N, N+1] or [B, N+1, N+1].
        """

        # necessary for multi-gpu models
        vol = K.reshape(inputs[0], (-1, *self.imshape))
        trf = K.reshape(inputs[1], (-1, *self.trfshape))


        # map transform across batch
        if self.single_transform:
            return tf.map_fn(lambda x: self._single_transform([x, trf[0, :]]), vol)
        else:
            return tf.map_fn(self._single_transform, [vol, trf], fn_output_signature=vol.dtype)

    def _single_transform(self, inputs):
        return transform(inputs[0],
                               inputs[1],
                               interp_method=self.interp_method,
                               fill_value=self.fill_value,
                               shift_center=self.shift_center,
                               shape=self.shape)


###############################################################################
# deformation utilities
###############################################################################


def transform(vol, loc_shift, interp_method='linear', fill_value=None,
              shift_center=True, shape=None):
    """Apply affine or dense transforms to images in N dimensions.

    Essentially interpolates the input ND tensor at locations determined by
    loc_shift. The latter can be an affine transform or dense field of location
    shifts in the sense that at location x we now have the data from x + dx, so
    we moved the data.

    Parameters:
        vol: tensor or array-like structure  of size vol_shape or
            (*vol_shape, C), where C is the number of channels.
        loc_shift: Affine transformation matrix of shape (N, N+1) or a shift
            volume of shape (*new_vol_shape, D) or (*new_vol_shape, C, D),
            where C is the number of channels, and D is the dimensionality
            D = len(vol_shape). If the shape is (*new_vol_shape, D), the same
            transform applies to all channels of the input tensor.
        interp_method: 'linear' or 'nearest'.
        fill_value: Value to use for points sampled outside the domain. If
            None, the nearest neighbors will be used.
        shift_center: Shift grid to image center when converting affine
            transforms to dense transforms. Assumes the input and output spaces are identical.
        shape: ND output shape used when converting affine transforms to dense
            transforms. Includes only the N spatial dimensions. If None, the
            shape of the input image will be used. Incompatible with `shift_center=True`.

    Returns:
        Tensor whose voxel values are the values of the input tensor
        interpolated at the locations defined by the transform.

    Notes:
        There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
        indexing. Due to inconsistencies in how some functions and layers handled xy-indexing, we
        removed it in favor of default ij-indexing to minimize the potential for confusion.

    Keywords:
        interpolation, sampler, resampler, linear, bilinear
    """
    if shape is not None and shift_center:
        raise ValueError('`shape` option incompatible with `shift_center=True`')

    # convert data type if needed
    ftype = tf.float32
    if not tf.is_tensor(vol) or not vol.dtype.is_floating:
        vol = tf.cast(vol, ftype)
    if not tf.is_tensor(loc_shift) or not loc_shift.dtype.is_floating:
        loc_shift = tf.cast(loc_shift, ftype)

    # convert affine to location shift (will validate affine shape)
    if is_affine_shape(loc_shift.shape):
        loc_shift = affine_to_dense_shift(loc_shift,
                                          shape=vol.shape[:-1] if shape is None else shape,
                                          shift_center=shift_center)

    # parse spatial location shape, including channels if available
    loc_volshape = loc_shift.shape[:-1]
    if isinstance(loc_volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
        loc_volshape = loc_volshape.as_list()

    # volume dimensions
    nb_dims = len(vol.shape) - 1
    is_channelwise = len(loc_volshape) == (nb_dims + 1)
    assert loc_shift.shape[-1] == nb_dims, \
        'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
        'with {}D transform'.format(nb_dims, vol.shape[:-1], loc_shift.shape[-1])

    # location should be mesh and delta
    mesh = ne.utils.volshape_to_meshgrid(loc_volshape, indexing='ij')  # volume mesh
    for d, m in enumerate(mesh):
        if m.dtype != loc_shift.dtype:
            mesh[d] = tf.cast(m, loc_shift.dtype)
    loc = [mesh[d] + loc_shift[..., d] for d in range(nb_dims)]

    # if channelwise location, then append the channel as part of the location lookup
    if is_channelwise:
        loc.append(mesh[-1])

    # test single
    return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)


def is_affine_shape(shape):
    """
    Determine whether the given shape (single-batch) represents an N-dimensional affine matrix of
    shape (M, N + 1), with `N in (2, 3)` and `M in (N, N + 1)`.

    Parameters:
        shape: Tuple or list of integers excluding the batch dimension.
    """
    if len(shape) == 2 and shape[-1] != 1:
        validate_affine_shape(shape)
        return True
    return False

def validate_affine_shape(shape):
    """
    Validate whether the input shape represents a valid affine matrix of shape (..., M, N + 1),
    where N is the number of dimensions, and M is N or N + 1. Throws an error if the shape is
    invalid.

    Parameters:
        shape: Tuple or list of integers.
    """
    ndim = shape[-1] - 1
    rows = shape[-2]
    if ndim not in (2, 3):
        raise ValueError(f'Affine matrix must be 2D or 3D, got {ndim}D')
    if rows not in (ndim, ndim + 1):
        raise ValueError(f'{ndim}D affine matrix must have {ndim} or {ndim + 1} rows, got {rows}.')


def affine_to_dense_shift(matrix, shape, shift_center=True, warp_right=None):
    """
    Convert N-dimensional (ND) matrix transforms to dense displacement fields.

    Algorithm:
        1. Build and (optionally) shift grid to center of image.
        2. Apply matrices to each index coordinate.
        3. Subtract grid.

    Parameters:
        matrix: Affine matrix of shape (..., M, N + 1), where M is N or N + 1. Can have any batch
            dimensions.
        shape: ND shape of the output space.
        shift_center: Shift grid to image center.
        warp_right: Right-compose the matrix transform with a displacement field of shape
            (..., *shape, N), with batch dimensions broadcastable to those of `matrix`.

    Returns:
        Dense shift (warp) of shape (..., *shape, N).

    Notes:
        There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
        indexing. Due to inconsistencies in how some functions and layers handled xy-indexing, we
        removed it in favor of default ij-indexing to minimize the potential for confusion.

    """
    if isinstance(shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        shape = shape.as_list()

    if not tf.is_tensor(matrix) or not matrix.dtype.is_floating:
        matrix = tf.cast(matrix, tf.float32)

    # check input shapes
    ndims = len(shape)
    if matrix.shape[-1] != (ndims + 1):
        matdim = matrix.shape[-1] - 1
        raise ValueError(f'Affine ({matdim}D) does not match target shape ({ndims}D).')
    validate_affine_shape(matrix.shape)

    # coordinate grid
    mesh = (tf.range(s, dtype=matrix.dtype) for s in shape)
    if shift_center:
        mesh = (m - 0.5 * (s - 1) for m, s in zip(mesh, shape))
    mesh = [tf.reshape(m, shape=(-1,)) for m in tf.meshgrid(*mesh, indexing='ij')]
    mesh = tf.stack(mesh, name='inverse')  # N x nb_voxels
    out = mesh

    # optionally right-compose with warp field
    if warp_right is not None:
        if not tf.is_tensor(warp_right) or warp_right.dtype != matrix.dtype:
            warp_right = tf.cast(warp_right, matrix.dtype)
        flat_shape = tf.concat((tf.shape(warp_right)[:-1 - ndims], (-1, ndims)), axis=0)
        warp_right = tf.reshape(warp_right, flat_shape)  # ... x nb_voxels x N
        out += tf.linalg.matrix_transpose(warp_right)  # ... x N x nb_voxels

    # compute locations, subtract grid to obtain shift
    out = matrix[..., :ndims, :-1] @ out + matrix[..., :ndims, -1:]  # ... x N x nb_voxels
    out = tf.linalg.matrix_transpose(out - mesh)  # ... x nb_voxels x N

    # restore shape
    shape = tf.concat((tf.shape(matrix)[:-2], (*shape, ndims)), axis=0)
    return tf.reshape(out, shape)  # ... x in_shape x N

class AffineToDenseShift(Layer):
    """
    Converts an affine transform to a dense shift transform.
    """

    def __init__(self, shape, shift_center=True, **kwargs):
        """
        Parameters:
            shape: Target shape of dense shift.
        """
        self.shape = shape
        self.ndims = len(shape)
        self.shift_center = shift_center
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shape': self.shape,
            'shift_center': self.shift_center,
        })
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.shape, self.ndims)

    def build(self, input_shape):
        validate_affine_shape(input_shape)

    def call(self, mat):
        """
        Parameters:
            mat: Affine matrices of shape (B, N, N+1).
        """
        return affine_to_dense_shift(mat, self.shape, shift_center=self.shift_center)


class VecInt(Layer):
    """
    Vector integration layer

    Enables vector integration via several methods (ode or quadrature for
    time-dependent vector fields and scaling-and-squaring for stationary fields)

    If you find this function useful, please cite:

      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    """

    def __init__(self,
                 method='ss',
                 int_steps=7,
                 out_time_pt=1,
                 ode_args=None,
                 odeint_fn=None,
                 **kwargs):
        """
        Parameters:
            method: Must be any of the methods in neuron.utils.integrate_vec.
            int_steps: Number of integration steps.
            out_time_pt: Time point at which to output if using odeint integration.

        Notes:
            There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
            indexing. Due to inconsistencies in how some functions and layers handled xy-indexing,
            we removed it in favor of default ij-indexing to minimize the potential for confusion.

        """
        # TODO: remove this block
        # load models saved with the `indexing` argument
        if 'indexing' in kwargs:
            del kwargs['indexing']
            warnings.warn('The `indexing` argument to VecInt no longer exists. If you loaded a '
                          'model, save it again to be able to load it in the future.')

        self.method = method
        self.int_steps = int_steps
        self.inshape = None
        self.out_time_pt = out_time_pt
        self.odeint_fn = odeint_fn  # if none then will use a tensorflow function
        self.ode_args = ode_args
        if ode_args is None:
            self.ode_args = {'rtol': 1e-6, 'atol': 1e-12}
        super(self.__class__, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'method': self.method,
            'int_steps': self.int_steps,
            'out_time_pt': self.out_time_pt,
            'ode_args': self.ode_args,
            'odeint_fn': self.odeint_fn,
        })
        return config

    def build(self, input_shape):
        # confirm built
        self.built = True

        trf_shape = input_shape
        if isinstance(input_shape[0], (list, tuple)):
            trf_shape = input_shape[0]
        self.inshape = trf_shape

        if trf_shape[-1] != len(trf_shape) - 2:
            raise Exception('transform ndims %d does not match expected ndims %d'
                            % (trf_shape[-1], len(trf_shape) - 2))

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        loc_shift = inputs[0]

        # necessary for multi-gpu models
        loc_shift = K.reshape(loc_shift, [-1, *self.inshape[1:]])
        if hasattr(inputs[0], '_keras_shape'):
            loc_shift._keras_shape = inputs[0]._keras_shape

        if len(inputs) > 1:
            assert self.out_time_pt is None, \
                'out_time_pt should be None if providing batch_based out_time_pt'

        # map transform across batch
        out = tf.map_fn(self._single_int,
                        [loc_shift] + inputs[1:],
                        fn_output_signature=loc_shift.dtype)
        if hasattr(inputs[0], '_keras_shape'):
            out._keras_shape = inputs[0]._keras_shape
        return out

    def _single_int(self, inputs):

        vel = inputs[0]
        out_time_pt = self.out_time_pt
        if len(inputs) == 2:
            out_time_pt = inputs[1]
        return integrate_vec(vel, method=self.method,
                                   nb_steps=self.int_steps,
                                   ode_args=self.ode_args,
                                   out_time_pt=out_time_pt,
                                   odeint_fn=self.odeint_fn)


def integrate_vec(vec, time_dep=False, method='ss', **kwargs):
    """
    Integrate (stationary of time-dependent) vector field (N-D Tensor) in tensorflow

    Aside from directly using tensorflow's numerical integration odeint(), also implements
    "scaling and squaring", and quadrature. Note that the diff. equation given to odeint
    is the one used in quadrature.

    Parameters:
        vec: the Tensor field to integrate.
            If vol_size is the size of the intrinsic volume, and vol_ndim = len(vol_size),
            then vector shape (vec_shape) should be
            [vol_size, vol_ndim] (if stationary)
            [vol_size, vol_ndim, nb_time_steps] (if time dependent)
        time_dep: bool whether vector is time dependent
        method: 'scaling_and_squaring' or 'ss' or 'ode' or 'quadrature'

        if using 'scaling_and_squaring': currently only supports integrating to time point 1.
            nb_steps: int number of steps. Note that this means the vec field gets broken
            down to 2**nb_steps. so nb_steps of 0 means integral = vec.

        if using 'ode':
            out_time_pt (optional): a time point or list of time points at which to evaluate
                Default: 1
            init (optional): if using 'ode', the initialization method.
                Currently only supporting 'zero'. Default: 'zero'
            ode_args (optional): dictionary of all other parameters for
                tf.contrib.integrate.odeint()

    Returns:
        int_vec: integral of vector field.
        Same shape as the input if method is 'scaling_and_squaring', 'ss', 'quadrature',
        or 'ode' with out_time_pt not a list. Will have shape [*vec_shape, len(out_time_pt)]
        if method is 'ode' with out_time_pt being a list.

    Todo:
        quadrature for more than just intrinsically out_time_pt = 1
    """

    if method not in ['ss', 'scaling_and_squaring', 'ode', 'quadrature']:
        raise ValueError("method has to be 'scaling_and_squaring' or 'ode'. found: %s" % method)

    if method in ['ss', 'scaling_and_squaring']:
        nb_steps = kwargs['nb_steps']
        assert nb_steps >= 0, 'nb_steps should be >= 0, found: %d' % nb_steps

        if time_dep:
            svec = K.permute_dimensions(vec, [-1, *range(0, vec.shape[-1] - 1)])
            assert 2**nb_steps == svec.shape[0], "2**nb_steps and vector shape don't match"

            svec = svec / (2**nb_steps)
            for _ in range(nb_steps):
                svec = svec[0::2] + tf.map_fn(transform, svec[1::2, :], svec[0::2, :])

            disp = svec[0, :]

        else:
            vec = vec / (2**nb_steps)
            for _ in range(nb_steps):
                vec += transform(vec, vec)
            disp = vec

    elif method == 'quadrature':
        # TODO: could output more than a single timepoint!
        nb_steps = kwargs['nb_steps']
        assert nb_steps >= 1, 'nb_steps should be >= 1, found: %d' % nb_steps

        vec = vec / nb_steps

        if time_dep:
            disp = vec[..., 0]
            for si in range(nb_steps - 1):
                disp += transform(vec[..., si + 1], disp)
        else:
            disp = vec
            for _ in range(nb_steps - 1):
                disp += transform(vec, disp)

    else:
        assert not time_dep, "odeint not implemented with time-dependent vector field"
        fn = lambda disp, _: transform(vec, disp)

        # process time point.
        out_time_pt = kwargs['out_time_pt'] if 'out_time_pt' in kwargs.keys() else 1
        out_time_pt = tf.cast(K.flatten(out_time_pt), tf.float32)
        len_out_time_pt = out_time_pt.get_shape().as_list()[0]
        assert len_out_time_pt is not None, 'len_out_time_pt is None :('
        # initializing with something like tf.zeros(1) gives a control flow issue.
        z = out_time_pt[0:1] * 0.0
        K_out_time_pt = K.concatenate([z, out_time_pt], 0)

        # enable a new integration function than tf.contrib.integrate.odeint
        odeint_fn = tf.contrib.integrate.odeint
        if 'odeint_fn' in kwargs.keys() and kwargs['odeint_fn'] is not None:
            odeint_fn = kwargs['odeint_fn']

        # process initialization
        if 'init' not in kwargs.keys() or kwargs['init'] == 'zero':
            disp0 = vec * 0  # initial displacement is 0
        else:
            raise ValueError('non-zero init for ode method not implemented')

        # compute integration with odeint
        if 'ode_args' not in kwargs.keys():
            kwargs['ode_args'] = {}
        disp = odeint_fn(fn, disp0, K_out_time_pt, **kwargs['ode_args'])
        disp = K.permute_dimensions(disp[1:len_out_time_pt + 1, :], [*range(1, len(disp.shape)), 0])

        # return
        if len_out_time_pt == 1:
            disp = disp[..., 0]

    return disp


def flowinverse(flow):
    flow0 = Lambda(lambda x: K.expand_dims(x[:, :, :, 0], axis=-1))(flow)
    flow1 = Lambda(lambda x: K.expand_dims(x[:, :, :, 1], axis=-1))(flow)

    flow0 = SpatialTransformer(name='inverse_stn1')([flow0, flow])
    flow1 = SpatialTransformer(name='inverse_stn2')([flow1, flow])

    flow_inverse = Concatenate()([flow0, flow1])
    flow_inverse = Lambda(lambda x: x * -1.)(flow_inverse)
    return flow_inverse


class MinMaxNormalizeLayer(Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(MinMaxNormalizeLayer, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs, training=None):
        # Compute min and max along the batch dimension
        min_values = tf.reduce_min(inputs, axis=0, keepdims=True)
        max_values = tf.reduce_max(inputs, axis=0, keepdims=True)

        # Apply min-max normalization
        normalized = (inputs - min_values) / (max_values - min_values + self.epsilon)
        return normalized