# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

#----------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

#----------------------------------------------------------------------------
# Paths.

data_dir = 'datasets'
training_data = 'GAN_classifier_datasets_train'
validation_data = 'GAN_classifier_datasets_val'
result_dir = 'models/%s' % training_data

#----------------------------------------------------------------------------
# TensorFlow options.

tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()        # Environment variables, set by the main program in train.py.

tf_config['graph_options.place_pruned_graph']   = True      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
tf_config['gpu_options.allow_growth']          	= False     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
env.CUDA_VISIBLE_DEVICES                       	= '0'       # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use.
env.TF_CPP_MIN_LOG_LEVEL                        = '1'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.

#----------------------------------------------------------------------------
# Official training configs, targeted mainly for CelebA-HQ.
# To run, comment/uncomment the lines as appropriate and launch train.py.

mode = 'postpool'#'postpool'#'predownscale'
switching_res = 4#128#64#32#16#8#4

latent_res = 4
rec_weight = 20.0
rec_G_weight = 0.1

desc        = 'pgan'                                        # Description string included in result subdir name.
random_seed = 1000                                          # Global random seed.
train       = EasyDict(func='train.train_classifier') # Options for main training func.
EG       	= EasyDict(func='networks.EG', fmap_base=1024, fmap_max=512, mode=mode, latent_res=latent_res, switching_res=switching_res)             # Options for encoder network in the image domain.
D_rec       = EasyDict(func='networks.D_patch', fmap_base=1024, fmap_max=512, latent_res=-1)             # Options for reconstruction-associated discriminator network.
EG_opt      = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for encoder and generator optimizer.
D_rec_opt   = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for reconstruction-associated discriminator optimizer.
EG_loss     = EasyDict(func='loss.EG_classification', rec_weight=rec_weight, rec_G_weight=rec_G_weight)        # Options for encoder and generator loss.
D_rec_loss  = EasyDict(func='loss.D_rec_wgangp')      # Options for reconstruction-associated discriminator loss.
sched       = EasyDict()                                    # Options for train.TrainingSchedule.
grid        = EasyDict(size='1080p', layout='row_per_class')       # Options for train.setup_snapshot_image_grid().

# Dataset (choose one).
desc += '-%s' % training_data; training_set = EasyDict(tfrecord_dir=training_data); validation_set = EasyDict(tfrecord_dir=validation_data); train.lr_mirror_augment = True; train.ud_mirror_augment = False; sched.lod_initial_resolution = 128; sched.lod_training_kimg = 25000; sched.lod_transition_kimg = 25000; train.total_kimg = 25000

# Conditioning & snapshot options.
desc += '-labels'; training_set.max_label_size = 'full'; validation_set.max_label_size = 'full' # conditioned on full label

# Config presets (choose one). Note: the official settings are optimal. It is not the larger batch size the better.
desc += '-preset-v2-1gpu'; num_gpus = 1; sched.minibatch_base = 32; sched.lrate_dict = {1024: 0.0015}

# Numerical precision (choose one).
desc += '-fp32'; sched.max_minibatch_per_gpu = {256: 16, 512: 8, 1024: 4}