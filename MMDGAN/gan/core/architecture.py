#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:34:47 2018

@author: mikolajbinkowski
"""
import tensorflow as tf
from core.ops import batch_norm, conv2d, deconv2d, linear, lrelu
from utils.misc import conv_sizes
# Generators

class Generator:
    def __init__(self, dim, c_dim, output_size, use_batch_norm, prefix='g_'):
        self.used = False
        self.dim = dim
        self.c_dim = c_dim
        self.output_size = output_size
        self.prefix = prefix
        if use_batch_norm:
            self.g_bn0 = batch_norm(name=prefix + 'bn0')
            self.g_bn1 = batch_norm(name=prefix + 'bn1')
            self.g_bn2 = batch_norm(name=prefix + 'bn2')
            self.g_bn3 = batch_norm(name=prefix + 'bn3')
            self.g_bn4 = batch_norm(name=prefix + 'bn4')
            self.g_bn5 = batch_norm(name=prefix + 'bn5')
        else:
            self.g_bn0 = lambda x: x
            self.g_bn1 = lambda x: x
            self.g_bn2 = lambda x: x
            self.g_bn3 = lambda x: x
            self.g_bn4 = lambda x: x
            self.g_bn5 = lambda x: x
            
    def __call__(self, seed, batch_size):
        with tf.variable_scope('generator') as scope:   
            if self.used:
                scope.reuse_variables()
            self.used = True
            return self.network(seed, batch_size)
        
    def network(self, seed, batch_size):
        pass

    
class DCGANGenerator(Generator):
    def network(self, seed, batch_size):
        s1, s2, s4, s8, s16 = conv_sizes(self.output_size, layers=4, stride=2)
        # 64, 32, 16, 8, 4 - for self.output_size = 64
        # default architecture
        # For Cramer: self.gf_dim = 64
        z_ = linear(seed, self.dim * 8 * s16 * s16, self.prefix + 'h0_lin') # project random noise seed and reshape
        
        h0 = tf.reshape(z_, [batch_size, s16, s16, self.dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0))
        
        h1 = deconv2d(h0, [batch_size, s8, s8, self.dim*4], name=self.prefix + 'h1')
        h1 = tf.nn.relu(self.g_bn1(h1))
                        
        h2 = deconv2d(h1, [batch_size, s4, s4, self.dim*2], name=self.prefix + 'h2')
        h2 = tf.nn.relu(self.g_bn2(h2))
        
        h3 = deconv2d(h2, [batch_size, s2, s2, self.dim*1], name=self.prefix + 'h3')
        h3 = tf.nn.relu(self.g_bn3(h3))
        
        h4 = deconv2d(h3, [batch_size, s1, s1, self.c_dim], name=self.prefix + 'h4')
        return tf.nn.sigmoid(h4)        


class DCGAN5Generator(Generator):
    def network(self, seed, batch_size):
        s1, s2, s4, s8, s16, s32 = conv_sizes(self.output_size, layers=5, stride=2)
        # project `z` and reshape
        z_= linear(seed, self.dim * 16 * s32 * s32, self.prefix + 'h0_lin')
        
        h0 = tf.reshape(z_, [-1, s32, s32, self.dim * 16])
        h0 = tf.nn.relu(self.g_bn0(h0))
        
        h1 = deconv2d(h0, [batch_size, s16, s16, self.dim*8], name=self.prefix + 'h1')
        h1 = tf.nn.relu(self.g_bn1(h1))
                        
        h2 = deconv2d(h1, [batch_size, s8, s8, self.dim*4], name=self.prefix + 'h2')
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, s4, s4, self.dim*2], name=self.prefix + 'h3')
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, s2, s2, self.dim], name=self.prefix + 'h4')
        h4 = tf.nn.relu(self.g_bn4(h4))                
        
        h5 = deconv2d(h4, [batch_size, s1, s1, self.c_dim], name=self.prefix + 'h5')
        return tf.nn.sigmoid(h5)


class ResNetGenerator(Generator):
    def network(self, seed, batch_size):
        from core.resnet import block, ops
        s1, s2, s4, s8, s16, s32 = conv_sizes(self.output_size, layers=5, stride=2)
        # project `z` and reshape
        z_= linear(seed, self.dim * 16 * s32 * s32, self.prefix + 'h0_lin')
        h0 = tf.reshape(z_, [-1, self.dim * 16, s32, s32]) # NCHW format
        h1 = block.ResidualBlock(self.prefix + 'res1', 16 * self.dim, 
                                 8 * self.dim, 3, h0, resample='up')
        h2 = block.ResidualBlock(self.prefix + 'res2', 8 * self.dim, 
                                 4 * self.dim, 3, h1, resample='up')
        h3 = block.ResidualBlock(self.prefix + 'res3', 4 * self.dim, 
                                 2 * self.dim, 3, h2, resample='up')
        h4 = block.ResidualBlock(self.prefix + 'res4', 2 * self.dim, 
                                 self.dim, 3, h3, resample='up')
        h4 = ops.batchnorm.Batchnorm('g_h4', [0, 2, 3], h4, fused=True)
        h4 = tf.nn.relu(h4)
#                h5 = lib.ops.conv2d.Conv2D('g_h5', dim, 3, 3, h4)
        h5 = tf.transpose(h4, [0, 2, 3, 1]) # NCHW to NHWC
        h5 = deconv2d(h5, [batch_size, s1, s1, self.c_dim], name='g_h5')
        return tf.nn.sigmoid(h5)


# Discriminator

class Discriminator:
    def __init__(self, dim, o_dim, use_batch_norm, prefix='d_'):
        self.dim = dim
        self.o_dim = o_dim 
        self.prefix = prefix
        self.used = False
        if use_batch_norm:
            self.d_bn0 = batch_norm(name=prefix + 'bn0')
            self.d_bn1 = batch_norm(name=prefix + 'bn1')
            self.d_bn2 = batch_norm(name=prefix + 'bn2')
            self.d_bn3 = batch_norm(name=prefix + 'bn3')
            self.d_bn4 = batch_norm(name=prefix + 'bn4')
            self.d_bn5 = batch_norm(name=prefix + 'bn5')
        else:
            self.d_bn0 = lambda x: x
            self.d_bn1 = lambda x: x
            self.d_bn2 = lambda x: x
            self.d_bn3 = lambda x: x
            self.d_bn4 = lambda x: x
            self.d_bn5 = lambda x: x
        
    def __call__(self, image, batch_size, return_layers=False):
        with tf.variable_scope("discriminator") as scope:
            if self.used:
                scope.reuse_variables()
            self.used = True
            
            layers = self.network(image, batch_size)
            
            if return_layers:
                return layers
            return layers['hF']
        
    def network(self, image, batch_size):
        pass

class DCGANDiscriminator(Discriminator):        
    def network(self, image, batch_size):
        o_dim = self.o_dim if (self.o_dim > 0) else 8 * self.dim
        h0 = lrelu(conv2d(image, self.dim, name=self.prefix + 'h0_conv')) 
        h1 = lrelu(self.d_bn1(conv2d(h0, self.dim * 2, name=self.prefix + 'h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.dim * 4, name=self.prefix + 'h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.dim * 8, name=self.prefix + 'h3_conv')))
        hF = linear(tf.reshape(h3, [batch_size, -1]), o_dim, self.prefix + 'h4_lin')
        
        return {'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'hF': hF}

class DCGAN5Discriminator(Discriminator):
    def network(self, image, batch_size):
        o_dim = self.o_dim if (self.o_dim > 0) else 16 * self.dim
        h0 = lrelu(conv2d(image, self.dim, name=self.prefix + 'h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.dim * 2, name=self.prefix + 'h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.dim * 4, name=self.prefix + 'h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.dim * 8, name=self.prefix + 'h3_conv')))
        h4 = lrelu(self.d_bn4(conv2d(h3, self.dim * 16, name=self.prefix + 'h4_conv')))
        hF = linear(tf.reshape(h4, [batch_size, -1]), o_dim, self.prefix + 'h6_lin')
        
        return {'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'hF': hF}        

class FullConvDiscriminator(Discriminator):
    def network(self, image, batch_size):
        h0 = lrelu(conv2d(image, self.dim, name=self.prefix + 'h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.dim * 2, name=self.prefix + 'h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.dim * 4, name=self.prefix + 'h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.dim * 8, name=self.prefix + 'h3_conv')))
        hF = lrelu(self.d_bn4(conv2d(h3, self.o_dim, name=self.prefix + 'hF_conv')))
        hF = tf.reshape(hF, [batch_size, -1])
        
        return {'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'hF': hF}

class ResNetDiscriminator(Discriminator):
    def network(self, image, batch_size):
        from core.resnet import block, ops
        image = tf.transpose(image, [0, 3, 1, 2]) # NHWC to NCHW
        
        h0 = lrelu(ops.conv2d.Conv2D(self.prefix + 'h0_conv', 3, self.dim, 
                                     3, image, he_init=False)) 
        h1 = block.ResidualBlock(self.prefix + 'res1', self.dim, 
                                 2 * self.dim, 3, h0, resample='down')
        h2 = block.ResidualBlock(self.prefix + 'res2', 2 * self.dim, 
                                 4 * self.dim, 3, h1, resample='down')
        h3 = block.ResidualBlock(self.prefix + 'res3', 4 * self.dim, 
                                 8 * self.dim, 3, h2, resample='down')
        h4 = block.ResidualBlock(self.prefix + 'res4', 8 * self.dim, 
                                 8 * self.dim, 3, h3, resample='down')
    
        hF = tf.reshape(h4, [-1, 4 * 4 * 8 * self.dim])
        hF = linear(hF, self.o_dim, self.prefix + 'h5_lin')
    
        return {'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'hF': hF}  

        
def get_networks(architecture):
    if architecture == 'dcgan':
        return DCGANGenerator, DCGANDiscriminator
    elif architecture == 'dcgan5':
        return DCGAN5Generator, DCGAN5Discriminator
    elif 'g_resnet5' in architecture:
        return ResNetGenerator, DCGAN5Discriminator
    elif architecture == 'resnet5':
        return ResNetGenerator, ResNetDiscriminator
    elif architecture == 'd_fullconv5':
        return DCGAN5Generator, FullConvDiscriminator
    raise ValueError('Wrong architecture: "%s"' % architecture)

