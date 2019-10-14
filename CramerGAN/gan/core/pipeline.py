#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:11:46 2018

@author: mikolajbinkowski
"""
import os, time, lmdb, io
import numpy as np
import tensorflow as tf
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
from utils import misc

class Pipeline:
    def __init__(self, output_size, c_dim, batch_size, data_dir, **kwargs):
        self.output_size = output_size
        self.c_dim = c_dim

#        data_dir = os.path.join(self.data_dir, self.dataset)
        self.batch_size = batch_size
        self.read_batch = max(4000, batch_size * 10)
        self.read_count = 0
        self.data_dir = data_dir
        self.shape = [self.read_batch, self.output_size, self.output_size, self.c_dim]
    
    def _transform(self, x):
        return x
    
    def connect(self):
        assert hasattr(self, 'single_sample'), 'Pipeline needs to have single_sample defined before connecting'
        self.single_sample.set_shape(self.shape)
        ims = tf.train.shuffle_batch([self.single_sample], self.batch_size,
                                    capacity=self.read_batch,
                                    min_after_dequeue=self.read_batch//8,
                                    num_threads=16,
                                    enqueue_many=len(self.shape) == 4)
        return self._transform(ims)
    

class LMDB(Pipeline):
    def __init__(self, *args, **kwargs):
        super(LMDB, self).__init__(*args, **kwargs)
        self.timer = kwargs.get('timer', None) 
        self.keys = []
        env = lmdb.open(self.data_dir, map_size=1099511627776, max_readers=100, readonly=True)
        with env.begin() as txn:
            cursor = txn.cursor()
            while cursor.next():
                self.keys.append(cursor.key())
        env.close()
        print('No. of records in lmdb database: %d' % len(self.keys))
        # tf queue for getting keys
        key_producer = tf.train.string_input_producer(self.keys, shuffle=True)
        single_key = key_producer.dequeue()
        self.single_sample = tf.py_func(self._get_sample_from_lmdb, [single_key], tf.float32)
        
        
    def _get_sample_from_lmdb(self, key, limit=None):
        if limit is None:
            limit = self.read_batch
        with tf.device('/cpu:0'):
            rc = self.read_count
            self.read_count += 1
            tt = time.time()
            self.timer(rc, 'lmdb: start reading chunk from database')
            ims = []
            db_count = 1
            while len(ims) < limit:
                env = lmdb.open(self.data_dir, map_size=1099511627776, max_readers=100, readonly=True)
                with env.begin(write=False) as txn:
                    cursor = txn.cursor()
                    cursor.set_key(key)
                    if not cursor.next():
                        cursor.first()
                    db_err = False
                    while (len(ims) < limit) and (not db_err):
                        try:
                            key, byte_arr = cursor.item()
                            byte_im = io.BytesIO(byte_arr)
                        #   byte_im.seek(0)
                            im = Image.open(byte_im)
                            ims.append(misc.center_and_scale(im, size=self.output_size))
                        except Exception as e:
                            self.timer(rc, 'lmdb error: ' + str(e))
                            self.timer(rc, 'lmdb open no. %d failed at key %s, with %d collected images' % (db_count, repr(key), len(ims)))
                            db_count += 1
                            db_err = True
                        if not cursor.next():
                            cursor.first()
                env.close()
            self.timer(rc, 'lmdb read time = %f' % (time.time() - tt))
            return np.asarray(ims, dtype=np.float32)       
     
        
    def constant_sample(self, size):
        choice = np.random.choice(self.keys, 1)[0]
        return self._get_sample_from_lmdb(choice, limit=size)


class TfRecords(Pipeline):
    def __init__(self, *args, **kwargs):  
        regex = os.path.join(self.data_dir, 'lsun-%d/bedroom_train_*' % self.output_size)
        filename_queue = tf.train.string_input_producer(tf.gfile.Glob(regex), num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
            'image/class/label': tf.FixedLenFeature([1], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
        })
        image = tf.image.decode_jpeg(features['image/encoded'])
        
        self.single_sample = tf.cast(image, tf.float32)/255.
        self.shape = [self.output_size, self.output_size, self.c_dim]


class JPEG_128(Pipeline):
    def __init__(self, *args, base_size=128, random_crop=9, **kwargs):
        super(JPEG_128, self).__init__(*args, **kwargs)
        #base_size = kwargs.get('base_size', 160)
        #random_crop = kwargs.get('random_crop', 9)
        files = glob(os.path.join(self.data_dir, '*.png'))

        filename_queue = tf.train.string_input_producer(files, shuffle=True)
        reader = tf.WholeFileReader()
        _, raw = reader.read(filename_queue)
        decoded = tf.image.decode_jpeg(raw, channels=self.c_dim) # HWC
        #bs = base_size + 2 * random_crop
        #cropped = tf.image.resize_image_with_crop_or_pad(decoded, bs, bs)
        #if random_crop > 0:
        #    cropped = tf.image.random_flip_left_right(cropped)
        #    cropped = tf.random_crop(cropped, [base_size, base_size, self.c_dim])
        self.single_sample = decoded
        self.shape = [base_size, base_size, self.c_dim]    
        
    def _transform(self, x):
        x = tf.image.resize_bilinear(x, (self.output_size, self.output_size))
        return tf.cast(x, tf.float32)/255.

class JPEG_64(Pipeline):
    def __init__(self, *args, base_size=64, random_crop=9, **kwargs):
        super(JPEG_64, self).__init__(*args, **kwargs)
        #base_size = kwargs.get('base_size', 160)
        #random_crop = kwargs.get('random_crop', 9)
        files = glob(os.path.join(self.data_dir, '*.png'))

        filename_queue = tf.train.string_input_producer(files, shuffle=True)
        reader = tf.WholeFileReader()
        _, raw = reader.read(filename_queue)
        decoded = tf.image.decode_jpeg(raw, channels=self.c_dim) # HWC
        #bs = base_size + 2 * random_crop
        #cropped = tf.image.resize_image_with_crop_or_pad(decoded, bs, bs)
        #if random_crop > 0:
        #    cropped = tf.image.random_flip_left_right(cropped)
        #    cropped = tf.random_crop(cropped, [base_size, base_size, self.c_dim])
        self.single_sample = decoded
        self.shape = [base_size, base_size, self.c_dim]    
        
    def _transform(self, x):
        x = tf.image.resize_bilinear(x, (self.output_size, self.output_size))
        return tf.cast(x, tf.float32)/255.


class Mnist(Pipeline):
    def __init__(self, *args, **kwargs):
        super(Mnist, self).__init__(*args, **kwargs)
        fd = open(os.path.join(self.data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)
    
        fd = open(os.path.join(self.data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)
    
        fd = open(os.path.join(self.data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)
    
        fd = open(os.path.join(self.data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)
    
        trY = np.asarray(trY)
        teY = np.asarray(teY)
    
        X = np.concatenate((trX, teX), axis=0).astype(np.float32) / 255.
        y = np.concatenate((trY, teY), axis=0)
    
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
    
        queue = tf.train.input_producer(tf.constant(X), shuffle=False)
        self.single_sample = queue.dequeue_many(self.read_batch)


class Cifar10(Pipeline):
    def __init__(self, *args, **kwargs):
        super(Cifar10, self).__init__(*args, **kwargs)
        categories = np.arange(10)
        batchesX, batchesY = [], []
        for batch in range(1,6):
            loaded = misc.unpickle(os.path.join(self.data_dir, 'data_batch_%d' % batch))
            idx = np.in1d(np.array(loaded['labels']), categories)
            batchesX.append(loaded['data'][idx].reshape(idx.sum(), 3, 32, 32))
            batchesY.append(np.array(loaded['labels'])[idx])
        trX = np.concatenate(batchesX, axis=0).transpose(0, 2, 3, 1)
        trY = np.concatenate(batchesY, axis=0)
        
        test = misc.unpickle(os.path.join(self.data_dir, 'test_batch'))
        idx = np.in1d(np.array(test['labels']), categories)
        teX = test['data'][idx].reshape(idx.sum(), 3, 32, 32).transpose(0, 2, 3, 1)
        teY = np.array(test['labels'])[idx]
    
        X = np.concatenate((trX, teX), axis=0).astype(np.float32) / 255.
        y = np.concatenate((trY, teY), axis=0)
    
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)

        queue = tf.train.input_producer(tf.constant(X), shuffle=False)
        self.single_sample = queue.dequeue_many(self.read_batch)
        

class GaussianMix(Pipeline):
    def __init__(self, *args, sample_dir='/', means=[.0, 3.0], stds=[1.0, .5], size=1000, **kwargs):
        super(GaussianMix, self).__init__(*args, **kwargs)
        from matplotlib import animation
        
        X_real = np.r_[
            np.random.normal(0,  1, size=size),
            np.random.normal(3, .5, size=size),
        ]   
        X_real = X_real.reshape(X_real.shape[0], 1, 1, 1)
        
        xlo = -5
        xhi = 7
        
        ax1 = plt.gca()
        fig = ax1.figure
        ax1.grid(False)
        ax1.set_yticks([], [])
        myhist(X_real.ravel(), color='r')
        ax1.set_xlim(xlo, xhi)
        ax1.set_ylim(0, 1.05)
        ax1._autoscaleXon = ax1._autoscaleYon = False
        
        wrtr = animation.writers['ffmpeg'](fps=20)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        wrtr.setup(fig=fig, outfile=os.path.join(sample_dir, 'train.mp4'), dpi=100)
        self.G_config = {'g_line': None,
                        'ax1': ax1,
                        'writer': wrtr,
                        'figure': ax1.figure}
        queue = tf.train.input_producer(tf.constant(X_real.astype(np.float32)), shuffle=False)
        self.single_sample = queue.dequeue_many(self.read_batch)

        
def myhist(X, ax=plt, bins='auto', **kwargs):
    hist, bin_edges = np.histogram(X, bins=bins)
    hist = hist / hist.max()
    return ax.plot(
        np.c_[bin_edges, bin_edges].ravel(),
        np.r_[0, np.c_[hist, hist].ravel(), 0],
        **kwargs
    )


def get_pipeline(dataset, info):
    if dataset == 'celebA' or dataset == 'lsun_bedroom_200k':        
        return JPEG_128 
    elif 'lsun' in dataset:
        if 'tf_records' in info:
            return TfRecords
        else:
            return LMDB
    elif dataset == 'mnist':
        return Mnist
    elif dataset == 'cifar10':
        return Cifar10
    elif dataset == 'GaussianMix':
        return GaussianMix
    elif dataset == 'tiny_imagenet':
        return JPEG_64
    else: 
        raise Exception('invalid dataset: %s' % dataset)         
