# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

# NOTE: Do not import any application-specific modules here!

#----------------------------------------------------------------------------

def lerp(a, b, t): return a + (b - a) * t
def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
def cset(cur_lambda, new_cond, new_lambda): return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    if kernel == 1:
        return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID', data_format='NCHW')
    else:
        x = tf.pad(x, paddings=[[0, 0],[0, 0],[kernel//2, kernel//2],[kernel//2, kernel//2]], mode='REFLECT')
        return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID', data_format='NCHW')

#----------------------------------------------------------------------------
# Valid convolutional layer.

def valid_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID', data_format='NCHW')

#----------------------------------------------------------------------------
# tanh valid convolutional layer.

def tanh_valid_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, tf.tanh(w), strides=[1,1,1,1], padding='VALID', data_format='NCHW')

#----------------------------------------------------------------------------
# Normalized valid convolutional layer.

def norm_valid_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=[1,2,3], keep_dims=True)), w / tf.sqrt(tf.reduce_sum(tf.square(w), axis=[0,1,2], keep_dims=True)), strides=[1,1,1,1], padding='VALID', data_format='NCHW')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Nearest-neighbor upscaling layer.

def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

#----------------------------------------------------------------------------
# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# upscale2d for RGB image by upsampling + Gaussian smoothing

gaussian_filter_up = tf.constant(list(np.float32([1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1])/256.0*4.0), dtype=tf.float32, shape=[5,5,1,1], name='GaussianFilterUp', verify_shape=False)

def upscale2d_rgb_Gaussian(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D_RGB_Gaussian'):
        for i in range(int(round(np.log2(factor)))):
            try:
                s = x.shape
                x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            except:
                s = tf.shape(x)
                x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.pad(x, paddings=[[0,0],[0,0],[0,0],[0,1],[0,0],[0,1]], mode='CONSTANT')
            x = tf.reshape(x, [-1, s[1], s[2]*2, s[3]*2])
            channel_list = []
            for j in range(3):
                z = tf.pad(x[:,j:j+1,:,:], paddings=[[0,0],[0,0],[2,2],[2,2]], mode='REFLECT')
                channel_list.append(tf.nn.conv2d(z, filter=gaussian_filter_up, strides=[1,1,1,1], padding='VALID', data_format='NCHW', name='GaussianConvUp'))
            x = tf.concat(channel_list, axis=1)
        return x

#----------------------------------------------------------------------------
# Box filter downscaling layer.

def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, factor, factor]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.

def conv2d_downscale2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# downscale2d for RGB image by Gaussian smoothing + downsampling

gaussian_filter_down = tf.constant(list(np.float32([1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1])/256.0), dtype=tf.float32, shape=[5,5,1,1], name='GaussianFilterDown', verify_shape=False)

def downscale2d_rgb_Gaussian(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D_RGB_Gaussian'):
        for i in range(int(round(np.log2(factor)))):
            channel_list = []
            for j in range(3):
                z = tf.pad(x[:,j:j+1,:,:], paddings=[[0,0],[0,0],[2,2],[2,2]], mode='REFLECT')
                channel_list.append(tf.nn.conv2d(z, filter=gaussian_filter_down, strides=[1,1,2,2], padding='VALID', data_format='NCHW', name='GaussianConvDown'))
            x = tf.concat(channel_list, axis=1)
        return x

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

#----------------------------------------------------------------------------
# Minibatch standard deviation.

def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

#----------------------------------------------------------------------------
# Encoder + generator network mapping from RGB image domain to latent domain.

def EG(
    images_in,                          # Input: Images [minibatch, channel, height, width].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 128,          # Input resolution. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    latent_res          = -1,           # Spatial dimension of the latent vectors.
    mode                = 'postpool',   # postpool means convolution first then pooling; predownscale means pooling first then convolution
    switching_res       = 4,            # The resolution to separate from subsequent convolution and subsequent pooling
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 0,            # Group size for the minibatch standard deviation layer, 0 = disable.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    normalize_latents   = False,        # Normalize latent vectors before feeding them to the network?
    use_pixelnorm       = False,        # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = False,        # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    resolution_log2 = int(np.log2(resolution))
    latent_res_log2 = 2 if latent_res == -1 else int(np.log2(latent_res))
    switching_res_log2 = int(np.log2(switching_res))
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    act = leaky_relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)

    # Building encoder blocks.
    def block_E(x, res): # res = latent_res_log2..resolution_log2
        with tf.variable_scope('E_%dx%d' % (2**res, 2**res)):
            if res > latent_res_log2:
                if mode == 'predownscale' and res <= switching_res_log2 or mode == 'postpool' and res > switching_res_log2:
                    with tf.variable_scope('Conv0'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                    if fused_scale:
                        with tf.variable_scope('Conv1_down'):
                            x = act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                    else:
                        with tf.variable_scope('Conv1'):
                            x = act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                        x = downscale2d(x)
                else:
                    if mode == 'predownscale':
                        x = downscale2d_rgb_Gaussian(x)
                    else:
                        x = downscale2d(x)
            else:
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv0'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                # fully connected
                if latent_res == -1:
                    with tf.variable_scope('Dense1'):
                        x = act(apply_bias(dense(x, fmaps=nf(res-1), use_wscale=use_wscale)))
                # fully convolutional
                else:
                    with tf.variable_scope('Conv1'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
            return x

    # Building generator blocks.
    def block_G(x, res): # res = 2..resolution_log2
        with tf.variable_scope('G_%dx%d' % (2**res, 2**res)):
            if res == latent_res_log2 and latent_res == -1:
                if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Dense0'):
                    x = dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
                    x = tf.reshape(x, [-1, nf(res-1), 4, 4])
                    x = PN(act(apply_bias(x)))
                with tf.variable_scope('Conv1'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
            elif res == latent_res_log2 and latent_res != -1:
                with tf.variable_scope('Conv0'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                with tf.variable_scope('Conv1'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
            elif res < resolution_log2:
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                else:
                    x = upscale2d(x)
                    with tf.variable_scope('Conv0'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                with tf.variable_scope('Conv1'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
            else:
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                else:
                    x = upscale2d(x)
                    with tf.variable_scope('Conv0'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                with tf.variable_scope('Conv1'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                with tf.variable_scope('Conv_toRGB'):
                    x = tf.tanh(apply_bias(conv2d(x, fmaps=num_channels, kernel=3, use_wscale=use_wscale)))
            return x

    def toLogits(images):
        with tf.variable_scope('Conv_fingerprints'):
            logits = valid_conv2d(images, fmaps=label_size, kernel=resolution, gain=1, use_wscale=use_wscale)
            #logits = tanh_valid_conv2d(images, fmaps=label_size, kernel=resolution, gain=1, use_wscale=use_wscale)
            #logits = norm_valid_conv2d(images, fmaps=label_size, kernel=resolution, gain=1, use_wscale=use_wscale)
        return tf.reshape(logits, shape=[-1,label_size])
    
    x = tf.identity(images_in)
    for res in range(resolution_log2, latent_res_log2-1, -1):
        x = block_E(x, res)
    for res in range(latent_res_log2, resolution_log2+1):
        x = block_G(x, res)
    images_out = tf.identity(x)
    fingerprints_out = tf.identity(images_in - images_out)
    logits_out = toLogits(fingerprints_out)

    assert images_out.dtype == tf.as_dtype(dtype) and fingerprints_out.dtype == tf.as_dtype(dtype) and logits_out.dtype == tf.as_dtype(dtype)
    return images_out, fingerprints_out, logits_out

#----------------------------------------------------------------------------
# Patch-based Discriminator network used in the paper without the auxiliary classifier.

def D_patch(
    images_in,                          # Input: Images [minibatch, channel, height, width].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 128,          # Input resolution. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    latent_res          = -1,           # Spatial dimension of the latent vectors.
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = False,        # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    resolution_log2 = int(np.log2(resolution))
    latent_res_log2 = 2 if latent_res == -1 else int(np.log2(latent_res))
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    act = leaky_relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)

    # Building blocks.
    def block(x, res): # res = latent_res_log2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res > latent_res_log2:
                with tf.variable_scope('Conv0'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        x = act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                else:
                    with tf.variable_scope('Conv1'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                    x = downscale2d(x)
            else:
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv0'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                # fully connected
                if latent_res == -1:
                    with tf.variable_scope('Dense1'):
                        x = act(apply_bias(dense(x, fmaps=nf(res-2), use_wscale=use_wscale)))
                    with tf.variable_scope('Dense2'):
                        x = apply_bias(dense(x, fmaps=1, gain=1, use_wscale=use_wscale))
                    x = tf.expand_dims(tf.expand_dims(x, axis=2), axis=3)
                # fully convolutional
                else:
                    with tf.variable_scope('Conv1'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=1, use_wscale=use_wscale)))
                    with tf.variable_scope('Conv2'):
                        x = apply_bias(conv2d(x, fmaps=1, gain=1, kernel=1, use_wscale=use_wscale))
            return x
    
    x = tf.identity(images_in)
    for res in range(resolution_log2, latent_res_log2-1, -1):
        x = block(x, res)
    scores_out = tf.identity(x)

    assert scores_out.dtype == tf.as_dtype(dtype)
    return scores_out