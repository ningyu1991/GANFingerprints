import sys
import numpy as np
import core
from utils.misc import pp, visualize

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("max_iteration", 150000, "Epoch to train [150000]")
flags.DEFINE_float("learning_rate", .0001, "Learning rate [.0001]")
flags.DEFINE_float("learning_rate_D", -1, "Learning rate for discriminator, if negative same as generator [-1]")
flags.DEFINE_boolean("MMD_lr_scheduler", True, "Wheather to use lr scheduler based on 3-sample test")
flags.DEFINE_float("decay_rate", .5, "Decay rate [.5]")
flags.DEFINE_float("gp_decay_rate", 1.0, "Decay rate [1.0]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("init", 0.1, "Initialization value [0.1]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [1000]")
flags.DEFINE_integer("real_batch_size", -1, "The size of batch images for real samples. If -1 then same as batch_size [-1]")
flags.DEFINE_integer("output_size", 128, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "celebA", "The name of the model fro saving puposes")
flags.DEFINE_string("name", "mmd_test", "The name of dataset [celebA, mnist, lsun, cifar10]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_mmd", "Directory name to save the checkpoints [checkpoint_mmd]")
flags.DEFINE_string("sample_dir", "samples_mmd", "Directory name to save the image samples [samples_mmd]")
flags.DEFINE_string("log_dir", "logs_mmd", "Directory name to save the image samples [logs_mmd]")
flags.DEFINE_string("data_dir", "./data", "Directory containing datasets [./data]")
flags.DEFINE_string("architecture", "dcgan", "The name of the architecture [dcgan, g-resnet5, dcgan5]")
flags.DEFINE_string("kernel", "", "The name of the architecture ['', 'mix_rbf', 'mix_rq', 'distance', 'dot', 'mix_rq_dot']")
flags.DEFINE_string("model", "mmd", "The model type [mmd, cramer, wgan_gp]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("is_demo", False, "For testing [False]")
flags.DEFINE_float("gradient_penalty", 0.0, "Use gradient penalty [0.0]")
flags.DEFINE_integer("threads", 64, "Upper limit for number of threads [np.inf]")
flags.DEFINE_integer("dsteps", 5, "Number of discriminator steps in a row [1] ")
flags.DEFINE_integer("gsteps", 1, "Number of generator steps in a row [1] ")
flags.DEFINE_integer("start_dsteps", 5, "Number of discrimintor steps in a row during first 20 steps and every 100th step" [1])
flags.DEFINE_integer("df_dim", 64, "Discriminator no of channels at first conv layer [64]")
flags.DEFINE_integer("dof_dim", 16, "No of discriminator output features [16]")
flags.DEFINE_integer("gf_dim", 64, "no of generator channels [64]")
flags.DEFINE_boolean("batch_norm", True, "Use of batch norm [False] (always False for discriminator if gradient_penalty > 0)")
flags.DEFINE_boolean("log", False, "Wheather to write log to a file in samples directory [True]")
flags.DEFINE_string("suffix", '', "For additional settings ['', '_tf_records']")
flags.DEFINE_boolean('compute_scores', False, "Compute scores [True]")
flags.DEFINE_float("gpu_mem", .9, "GPU memory fraction limit [0.9]")
flags.DEFINE_float("L2_discriminator_penalty", 0.0, "L2 penalty on discriminator features [0.0]")
flags.DEFINE_integer("no_of_samples", 100000, "number of samples to produce")
flags.DEFINE_boolean("print_pca", False, "")
flags.DEFINE_integer("save_layer_outputs", 0, "Wheather to save_layer_outputs. If == 2, saves outputs at exponential steps: 1, 2, 4, ..., 512 and every 1000. [0, 1, 2]")
flags.DEFINE_string("output_dir_of_test_samples", 'samples_mmd', "Output directory for testing samples")
flags.DEFINE_integer("random_seed", 0, "Random seed")
FLAGS = flags.FLAGS

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

def create_session(config_dict=dict(), force_as_default=False):
    config = tf.ConfigProto()
    for key, value in config_dict.items():
        fields = key.split('.')
        obj = config
        for field in fields[:-1]:
            obj = getattr(obj, field)
        setattr(obj, fields[-1], value)
    session = tf.Session(config=config)
    if force_as_default:
        session._default_session = session.as_default()
        session._default_session.enforce_nesting = False
        session._default_session.__enter__()
    return session

if tf.get_default_session() is None:
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(np.random.randint(1 << 31))
    tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
    tf_config['graph_options.place_pruned_graph']   = True      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
    tf_config['gpu_options.allow_growth']          = False     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
    create_session(tf_config, force_as_default=True)

def main(_):
    pp.pprint(FLAGS.__flags)
        
    if FLAGS.threads < np.inf:
        sess_config = tf.ConfigProto(intra_op_parallelism_threads=FLAGS.threads)
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_mem
        
    else:
        sess_config = tf.ConfigProto()
    if 'mmd' in FLAGS.model:
        from core.model import MMD_GAN as Model
    elif FLAGS.model == 'wgan_gp':
        from core.wgan_gp import WGAN_GP as Model
    elif 'cramer' in FLAGS.model:
        from core.cramer import Cramer_GAN as Model

        
    with tf.Session(config=sess_config) as sess:
        if FLAGS.dataset == 'mnist':
            gan = Model(sess, config=FLAGS, batch_size=FLAGS.batch_size, output_size=28, c_dim=1,
                        data_dir=FLAGS.data_dir)
        elif FLAGS.dataset == 'cifar10':
            gan = Model(sess, config=FLAGS, batch_size=FLAGS.batch_size, output_size=32, c_dim=3,
                        data_dir=FLAGS.data_dir)
        elif FLAGS.dataset in  ['celebA', 'lsun']:
            gan = Model(sess, config=FLAGS, batch_size=FLAGS.batch_size, output_size=FLAGS.output_size, c_dim=3,
                        data_dir=FLAGS.data_dir)
        else:
            gan = Model(sess, config=FLAGS, batch_size=FLAGS.batch_size, 
                        output_size=FLAGS.output_size, c_dim=FLAGS.c_dim,
                        data_dir=FLAGS.data_dir)
            
        if FLAGS.is_train:
            gan.train()
        elif FLAGS.print_pca:
            gan.print_pca()
        elif FLAGS.visualize:
            gan.load_checkpoint()
            visualize(sess, gan, FLAGS, 2)
        else:
            gan.get_samples(FLAGS.no_of_samples, layers=[-1])


        if FLAGS.log:
            sys.stdout = gan.old_stdout
            gan.log_file.close()
        gan.sess.close()
        
if __name__ == '__main__':
    tf.app.run()
