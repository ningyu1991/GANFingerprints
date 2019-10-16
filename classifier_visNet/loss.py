# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to

# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def EG_classification(EG, D_rec, reals_orig, labels, rec_weight, rec_G_weight):

    recs_out, fingerprints_out, logits_out = EG.get_output_for(reals_orig)
    real_class_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits_out, dim=1)
    real_class_loss = tfutil.autosummary('Loss/real_class_loss', real_class_loss)
    loss = tf.identity(real_class_loss)

    if rec_weight > 0.0:
        rec_loss = tf.reduce_mean(tf.abs(recs_out - reals_orig), axis=[1,2,3])
        rec_loss *= rec_weight
        rec_loss = tfutil.autosummary('Loss/rec_loss', rec_loss)
        loss += rec_loss

    if rec_G_weight > 0.0:
        rec_scores_out = fp32(D_rec.get_output_for(recs_out))
        rec_G_loss = tf.reduce_mean(-rec_scores_out, axis=[1,2,3])
        rec_G_loss *= rec_G_weight
        rec_G_loss = tfutil.autosummary('Loss/rec_G_loss', rec_G_loss)
        loss += rec_G_loss

    return loss

#----------------------------------------------------------------------------
# Reconstruction-associated discriminator loss function used in the paper (WGAN-GP).

def D_rec_wgangp(EG, D_rec, D_rec_opt, minibatch_size, reals_orig,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0):     # Target value for gradient magnitudes.

    # reconstructed realism
    recs_out, fingerprints_out, logits_out = EG.get_output_for(reals_orig)
    rec_scores_out = fp32(D_rec.get_output_for(recs_out))
    real_scores_out = fp32(D_rec.get_output_for(reals_orig))
    rec_D_loss = tf.reduce_mean(rec_scores_out - real_scores_out, axis=[1,2,3])
    rec_D_loss = tfutil.autosummary('Loss/rec_D_loss', rec_D_loss)
    loss = tf.identity(rec_D_loss)

    # gradient penalty
    with tf.name_scope('rec_GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=recs_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals_orig, recs_out.dtype), recs_out, mixing_factors)
        mixed_scores_out = fp32(D_rec.get_output_for(mixed_images_out))
        mixed_loss = D_rec_opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = D_rec_opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        rec_gradient_penalty = tf.square(mixed_norms - wgan_target)
        rec_gradient_penalty *= (wgan_lambda / (wgan_target**2))
        rec_gradient_penalty = tfutil.autosummary('Loss/rec_gradient_penalty', rec_gradient_penalty)
    loss += rec_gradient_penalty

    # calibration penalty
    with tf.name_scope('rec_EpsilonPenalty'):
        rec_epsilon_penalty = tf.reduce_mean(tf.square(real_scores_out), axis=[1,2,3]) * wgan_epsilon
        rec_epsilon_penalty = tfutil.autosummary('Loss/rec_epsilon_penalty', rec_epsilon_penalty)
    loss += rec_epsilon_penalty

    return loss