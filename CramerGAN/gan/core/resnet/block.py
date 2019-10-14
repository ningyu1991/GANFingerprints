"""
Based on https://github.com/igul222/improved_wgan_training/blob/master/gan_64x64.py.
"""
import functools
import tensorflow as tf
from core.resnet.ops import conv2d, batchnorm, layernorm


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = conv2d.Conv2D
        conv_1        = functools.partial(conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.BN1', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output


def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

    
def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output
    
    
def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output
    

def Normalize(name, axes, inputs):
    if ('d_' in name):# and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return batchnorm.Batchnorm(name,axes,inputs,fused=True)
