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

def C_classification(C_im, reals_orig, labels):

    with tf.name_scope('ClassificationPenalty'):
        real_labels_out = fp32(C_im.get_output_for(reals_orig))
        real_class_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out, dim=1)
        real_class_loss = tfutil.autosummary('Loss/real_class_loss', real_class_loss)
    loss = tf.identity(real_class_loss)

    return loss