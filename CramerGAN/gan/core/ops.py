from tensorflow.python.framework import ops
from utils.misc import variable_summaries
from .mmd import _eps, tf


class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                       tf.get_variable_scope().name)
        has_summary = any([('w' in v.op.name) for v in scope_vars])
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        
        if not has_summary:
            variable_summaries({'W': w, 'b': biases})    
        
        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                       tf.get_variable_scope().name)
        has_summary = any([('w' in v.op.name) for v in scope_vars])
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        
        if not has_summary:
            variable_summaries({'W': w, 'b': biases})
            
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def linear(input_, output_size, name="Linear", stddev=0.01, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    
    with tf.variable_scope(name):
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                       tf.get_variable_scope().name)
        has_summary = any([('Matrix' in v.op.name) for v in scope_vars])
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        
        if not has_summary:
            variable_summaries({'W': matrix, 'b': bias})
        
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


class linear_n:
    def __init__(self, input_, output_size, scope=None, stddev=0.1,
                 bias_start=0., train_scale=False):
        shape = input_.get_shape().as_list()

        with tf.variable_scope(scope or "Linear"):
            self.matrix = tf.get_variable(
                "Matrix", [shape[1], output_size], tf.float32,
                tf.random_normal_initializer(stddev=stddev))
            self.scale = tf.get_variable(
                "scale", [output_size], tf.float32, tf.constant_initializer(1.0),
                trainable=train_scale)
            self.b = tf.get_variable(
                "bias", [output_size], tf.float32, tf.constant_initializer(bias_start))
            self.scale_ = tf.get_variable(
                "scale_", [output_size], tf.float32, tf.constant_initializer(1.0))

        self.W = self.matrix * (self.scale/tf.sqrt(tf.reduce_sum(tf.square(self.matrix),0)))
        self.out = self.output(input_)

    def output(self, inp):
        return tf.matmul(inp, self.W) + self.b

    def init_op(self):
        mean = tf.reduce_mean(self.out, 0)
        stdv = tf.sqrt(tf.reduce_mean(tf.square(self.out), 0))
        self.out = (self.out - mean)/stdv
        scale_update_op = tf.assign(self.scale, self.scale/stdv)
        b_update_op = tf.assign(self.b, -mean/stdv)
        return tf.group(*[scale_update_op, b_update_op])

    def l2_normalize_op(self):
        self.W = self.W * (self.scale_ / tf.sqrt(
            1e-6 + tf.reduce_sum(tf.square(self.W), 0)))


def safer_norm(tensor, axis=None, keep_dims=False, epsilon=_eps):
    sq = tf.square(tensor)
    squares = tf.reduce_sum(sq, axis=axis, keep_dims=keep_dims)
    return tf.sqrt(squares + epsilon)


def sq_sum(t, name=None):
    "The squared Frobenius-type norm of a tensor, sum(t ** 2)."
    with tf.name_scope(name, "SqSum", [t]):
        t = tf.convert_to_tensor(t, name='t')
        return 2 * tf.nn.l2_loss(t)


def dot(x, y, name=None):
    "The dot product of two vectors x and y."
    with tf.name_scope(name, "Dot", [x, y]):
        x = tf.convert_to_tensor(x, name='x')
        y = tf.convert_to_tensor(y, name='y')

        x.get_shape().assert_has_rank(1)
        y.get_shape().assert_has_rank(1)

        return tf.squeeze(tf.matmul(tf.expand_dims(x, 0), tf.expand_dims(y, 1)))
