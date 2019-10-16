# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licen sed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import time
import numpy as np
import tensorflow as tf

import config
import tfutil
import dataset
import misc

import argparse

#----------------------------------------------------------------------------
# Choose the size and contents of the image snapshot grids that are exported
# periodically during training.

def setup_snapshot_image_grid(training_set, drange_net, grid_size=None,
    size    = '1080p',      # '1080p' = to be viewed on 1080p display, '4k' = to be viewed on 4k display.
    layout  = 'random'):    # 'random' = grid contents are selected randomly, 'row_per_class' = each row corresponds to one class label.

    # Select size.
    if grid_size is None:
        if size == '1080p':
            gw = np.clip(1920 // training_set.shape[2], 3, 32)
            gh = np.clip(1080 // training_set.shape[1], 2, 32)
        if size == '4k':
            gw = np.clip(3840 // training_set.shape[2], 7, 32)
            gh = np.clip(2160 // training_set.shape[1], 4, 32)
    else:
        gw = grid_size[0]
        gh = grid_size[1]

    # Fill in reals and labels.
    reals = np.zeros([gw * gh] + training_set.shape, dtype=np.float32)
    labels = np.zeros([gw * gh, training_set.label_size], dtype=training_set.label_dtype)
    for idx in range(gw * gh):
        x = idx % gw; y = idx // gw
        while True:
            real, label = training_set.get_minibatch_np(1)
            real = real.astype(np.float32)
            real = misc.adjust_dynamic_range(real, training_set.dynamic_range, drange_net)
            if layout == 'row_per_class' and training_set.label_size > 0:
                if label[0, y % training_set.label_size] == 0.0:
                    continue
            reals[idx] = real[0]
            labels[idx] = label[0]
            break

    return (gw, gh), reals, labels

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, lod, lr_mirror_augment, ud_mirror_augment, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
        if lr_mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        if ud_mirror_augment:
            with tf.name_scope('udMirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[2]))
        with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
            s = tf.shape(x)
            y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
            y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
            y = tf.tile(y, [1, 1, 1, 2, 1, 2])
            y = tf.reshape(y, [-1, s[1], s[2], s[3]])
            x_fade = tfutil.lerp(x, y, lod - tf.floor(lod))
            x_orig = tf.identity(x)
        with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
            s = tf.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            x_fade = tf.reshape(x_fade, [-1, s[1], s[2], 1, s[3], 1])
            x_fade = tf.tile(x_fade, [1, 1, 1, factor, 1, factor])
            x_fade = tf.reshape(x_fade, [-1, s[1], s[2] * factor, s[3] * factor])
            x_orig = tf.reshape(x_orig, [-1, s[1], s[2], 1, s[3], 1])
            x_orig = tf.tile(x_orig, [1, 1, 1, factor, 1, factor])
            x_orig = tf.reshape(x_orig, [-1, s[1], s[2] * factor, s[3] * factor])
        return x_fade, x_orig

#----------------------------------------------------------------------------
# Class for evaluating and storing the values of time-varying training parameters.

class TrainingSchedule:
    def __init__(
        self,
        cur_nimg,
        training_set,
        lod_initial_resolution  = 128,        # Image resolution used at the beginning.
        lod_training_kimg       = 1500,     # Thousands of real images to show before doubling the resolution.
        lod_transition_kimg     = 1500,     # Thousands of real images to show when fading in new layers.
        minibatch_base          = 16,       # Maximum minibatch size, divided evenly among GPUs.
        minibatch_dict          = {},       # Resolution-specific overrides.
        max_minibatch_per_gpu   = {},       # Resolution-specific maximum minibatch size per GPU.
        lrate_base              = 0.001,    # Learning rate for AutoEncoder.
        lrate_dict              = {},       # Resolution-specific overrides.
        tick_kimg_base          = 1,        # Default interval of progress snapshots.
        tick_kimg_dict          = {}):      # Resolution-specific overrides.

        # Training phase.
        self.kimg = cur_nimg / 1000.0
        phase_dur = lod_training_kimg + lod_transition_kimg
        phase_idx = int(np.floor(self.kimg / phase_dur)) if phase_dur > 0 else 0
        phase_kimg = self.kimg - phase_idx * phase_dur

        # Level-of-detail and resolution.
        self.lod = training_set.resolution_log2
        self.lod -= np.floor(np.log2(lod_initial_resolution))
        self.lod -= phase_idx
        if lod_transition_kimg > 0:
            self.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        self.lod = max(self.lod, 0.0)
        self.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(self.lod)))

        # Minibatch size.
        self.minibatch = minibatch_dict.get(self.resolution, minibatch_base)
        self.minibatch -= self.minibatch % config.num_gpus
        if self.resolution in max_minibatch_per_gpu:
            self.minibatch = min(self.minibatch, max_minibatch_per_gpu[self.resolution] * config.num_gpus)

        # Other parameters.
        self.lrate = lrate_dict.get(self.resolution, lrate_base)
        self.tick_kimg = tick_kimg_dict.get(self.resolution, tick_kimg_base)

#----------------------------------------------------------------------------
# Main training script.
# To run, comment/uncomment appropriate lines in config.py and launch train.py.

def train_classifier(
    smoothing               = 0.999,        # Exponential running average of encoder weights.
    minibatch_repeats       = 4,            # Number of minibatches to run before adjusting training parameters.
    reset_opt_for_new_lod   = True,         # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 100000,        # Total length of the training, measured in thousands of real images.
    lr_mirror_augment       = True,        # Enable mirror augment?
    ud_mirror_augment       = False,        # Enable up-down mirror augment?
    drange_net              = [-1,1],       # Dynamic range used when feeding image data to the networks.
    accuracy_snapshot_ticks    = 10,           # How often to export image snapshots?
    save_tf_graph           = False,        # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False):       # Include weight histograms in the tfevents file?

    maintenance_start_time = time.time()
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.training_set)
    validation_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.validation_set)
    network_snapshot_ticks = total_kimg // 100 # How often to export network snapshots?

    # Construct networks.
    with tf.device('/gpu:0'):
        try:
            network_pkl = misc.locate_network_pkl()
            resume_kimg, resume_time = misc.resume_kimg_time(network_pkl)
            print('Loading networks from "%s"...' % network_pkl)
            C_im = misc.load_pkl(network_pkl)
        except:
            print('Constructing networks...')
            resume_kimg = 0.0
            resume_time = 0.0
            C_im = tfutil.Network('C_im', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **config.C_im)
    C_im.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'):
        lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // config.num_gpus
        reals, labels   = training_set.get_minibatch_tf()
        reals_split     = tf.split(reals, config.num_gpus)
        labels_split    = tf.split(labels, config.num_gpus)
    C_opt = tfutil.Optimizer(name='TrainC', learning_rate=lrate_in, **config.C_opt)
    for gpu in range(config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            C_im_gpu = C_im if gpu == 0 else C_im.clone(C_im.name + '_shadow_%d' % gpu)
            reals_fade_gpu, reals_orig_gpu = process_reals(reals_split[gpu], lod_in, lr_mirror_augment, ud_mirror_augment, training_set.dynamic_range, drange_net)
            labels_gpu = labels_split[gpu]
            with tf.name_scope('C_loss'):
                C_loss = tfutil.call_func_by_name(C_im=C_im_gpu, reals_orig=reals_orig_gpu, labels=labels_gpu, **config.C_loss)
            C_opt.register_gradients(tf.reduce_mean(C_loss), C_im_gpu.trainables)
    C_train_op = C_opt.apply_updates()

    print('Setting up snapshot image grid...')
    grid_size, train_reals, train_labels = setup_snapshot_image_grid(training_set, drange_net, [5000, 1], **config.grid)
    grid_size, val_reals, val_labels = setup_snapshot_image_grid(validation_set, drange_net, [5000, 1], **config.grid)
    sched = TrainingSchedule(total_kimg * 1000, training_set, **config.sched)

    train_logits = C_im.run(train_reals, minibatch_size=sched.minibatch//config.num_gpus)
    train_preds = np.argmax(train_logits, axis=1)
    train_gt = np.argmax(train_labels, axis=1)
    train_acc = np.float32(np.sum(train_gt==train_preds)) / np.float32(len(train_gt))
    print('Training Accuracy = %f' % train_acc)

    val_logits = C_im.run(val_reals, minibatch_size=sched.minibatch//config.num_gpus)
    val_preds = np.argmax(val_logits, axis=1)
    val_gt = np.argmax(val_labels, axis=1)
    val_acc = np.float32(np.sum(val_gt==val_preds)) / np.float32(len(val_gt))
    print('Val Accuracy = %f' % val_acc)

    print('Setting up result dir...')
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    misc.save_image_grid(train_reals[:120,:,:,:], os.path.join(result_subdir, 'train_reals.png'), drange=drange_net, grid_size=[15,8])
    misc.save_image_grid(val_reals[:120,:,:,:], os.path.join(result_subdir, 'val_reals.png'), drange=drange_net, grid_size=[15,8])

    summary_log = tf.summary.FileWriter(result_subdir)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        C_im.setup_weight_histograms()

    print('Training...')
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    train_start_time = tick_start_time - resume_time
    prev_lod = -1.0
    while cur_nimg < total_kimg * 1000:

        # Choose training parameters and configure training ops.
        sched = TrainingSchedule(cur_nimg, training_set, **config.sched)
        training_set.configure(sched.minibatch, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                C_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops.
        for repeat in range(minibatch_repeats):
            tfutil.run([C_train_op], {lod_in: sched.lod, lrate_in: sched.lrate, minibatch_in: sched.minibatch})
            cur_nimg += sched.minibatch

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            total_time = cur_time - train_start_time
            maintenance_time = tick_start_time - maintenance_start_time
            maintenance_start_time = cur_time

            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f resolution %-4d minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %.1f' % (
                tfutil.autosummary('Progress/tick', cur_tick),
                tfutil.autosummary('Progress/kimg', cur_nimg / 1000.0),
                tfutil.autosummary('Progress/lod', sched.lod),
                tfutil.autosummary('Progress/resolution', sched.resolution),
                tfutil.autosummary('Progress/minibatch', sched.minibatch),
                misc.format_time(tfutil.autosummary('Timing/total_sec', total_time)),
                tfutil.autosummary('Timing/sec_per_tick', tick_time),
                tfutil.autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                tfutil.autosummary('Timing/maintenance_sec', maintenance_time)))
            tfutil.autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            tfutil.autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))
            tfutil.save_summaries(summary_log, cur_nimg)

            # Print accuracy.
            if cur_tick % accuracy_snapshot_ticks == 0 or done:

                train_logits = C_im.run(train_reals, minibatch_size=sched.minibatch//config.num_gpus)
                train_preds = np.argmax(train_logits, axis=1)
                train_gt = np.argmax(train_labels, axis=1)
                train_acc = np.float32(np.sum(train_gt==train_preds)) / np.float32(len(train_gt))
                print('Training Accuracy = %f' % train_acc)

                val_logits = C_im.run(val_reals, minibatch_size=sched.minibatch//config.num_gpus)
                val_preds = np.argmax(val_logits, axis=1)
                val_gt = np.argmax(val_labels, axis=1)
                val_acc = np.float32(np.sum(val_gt==val_preds)) / np.float32(len(val_gt))
                print('Val Accuracy = %f' % val_acc)
            
            if cur_tick % network_snapshot_ticks == 0 or done:
                misc.save_pkl(C_im, os.path.join(result_subdir, 'network-snapshot-%06d.pkl' % (cur_nimg // 1000)))

            # Record start time of the next tick.
            tick_start_time = time.time()

    # Write final results.
    misc.save_pkl(C_im, os.path.join(result_subdir, 'network-final.pkl'))
    summary_log.close()
    open(os.path.join(result_subdir, '_training-done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Main entry point.
# Calls the function indicated in config.py.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', type=str, default=' ')
    #------------------- training arguments -------------------
    parser.add_argument('--training_data_dir', type=str, default=' ') # The prepared training dataset directory that can be efficiently called by the code
    parser.add_argument('--validation_data_dir', type=str, default=' ') # The prepared validation dataset directory that can be efficiently called by the code
    parser.add_argument('--out_model_dir', type=str, default=' ') # The output directory containing trained models, training configureation, training log, and training snapshots
    parser.add_argument('--training_seed', type=int, default=1000) # The random seed that differentiates training instances
    #------------------- image generation arguments -------------------
    parser.add_argument('--model_path', type=str, default=' ') # The pre-trained GAN model
    parser.add_argument('--testing_data_path', type=str, default=' ') # The path of testing image file or the directory containing a collection of testing images

    args = parser.parse_args()
    if args.app == 'train':
        assert args.training_data_dir != ' ' and args.out_model_dir != ' '
        misc.init_output_logging()
        np.random.seed(args.training_seed)
        print('Initializing TensorFlow...')
        os.environ.update(config.env)
        tfutil.init_tf(config.tf_config)
        if args.training_data_dir[-1] == '/':
            args.training_data_dir = args.training_data_dir[:-1]
        idx = args.training_data_dir.rfind('/')
        config.data_dir = args.training_data_dir[:idx]
        config.training_set = config.EasyDict(tfrecord_dir=args.training_data_dir[idx+1:], max_label_size='full')
        if args.validation_data_dir[-1] == '/':
            args.validation_data_dir = args.validation_data_dir[:-1]
        idx = args.validation_data_dir.rfind('/')
        config.validation_set = config.EasyDict(tfrecord_dir=args.validation_data_dir[idx+1:], max_label_size='full')
        app = config.EasyDict(func='run.train_classifier', lr_mirror_augment=True, ud_mirror_augment=False, total_kimg=100000)
        config.result_dir = args.out_model_dir
    elif args.app == 'test':
        assert args.model_path != ' ' and args.testing_data_path != ' '
        misc.init_output_logging()
        print('Initializing TensorFlow...')
        os.environ.update(config.env)
        tfutil.init_tf(config.tf_config)
        app = config.EasyDict(func='util_scripts.classify', model_path=args.model_path, testing_data_path=args.testing_data_path)

    tfutil.call_func_by_name(**app)