# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import glob
import argparse
import numpy as np
import tensorflow as tf
import PIL.Image

import scipy.ndimage

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    exit(1)

#----------------------------------------------------------------------------

class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=10, Gaussian_down=0):
        self.tfrecord_dir       = tfrecord_dir
        self.tfr_prefix         = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.shape              = None
        self.resolution_log2    = None
        self.tfr_writers        = []
        self.print_progress     = print_progress
        self.progress_interval  = progress_interval
        self.Gaussian_down      = Gaussian_down
        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert(os.path.isdir(self.tfrecord_dir))
        
    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self): # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(self, img):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2**self.resolution_log2
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            #for lod in range(self.resolution_log2 - 1):
            for lod in [0]:
                tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                if self.Gaussian_down:
                    img = scipy.ndimage.convolve(img, gaussian_filter[np.newaxis, :, :], mode='mirror')[:, ::2, ::2]
                else:
                    img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))
            
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

#----------------------------------------------------------------------------

def data_preparation(image_dir, tfrecord_dir, resolution=128, shuffle=1, export_labels=1, percentage_samples=100):
    print('Loading images from "%s"' % image_dir)
    sources_dir = os.listdir(image_dir)
    image_filenames = []
    for source_dir in sources_dir:
        image_filenames_temp = sorted(glob.glob(os.path.join(image_dir, source_dir, '*.png')))
        image_filenames += image_filenames_temp[:int(float(len(image_filenames_temp))*float(percentage_samples)/100.0)]
    if len(image_filenames) == 0:
        error('No input images found')
        
    img = np.asarray(PIL.Image.open(image_filenames[0]))
    size = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')
    
    with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
        for idx in range(order.size):
            img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
            if channels == 1:
                img = img[:, :, np.newaxis] # HW => HWC
            img = PIL.Image.fromarray(img, 'RGB')
            img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
            img = np.asarray(img)
            img = img.transpose(2, 0, 1) # HWC => CHW
            tfr.add_image(img)
        if export_labels:
            labels = []
            for file in image_filenames:
                name_list = file.split('/')
                file_source = name_list[-2]
                for label, source in enumerate(sources_dir):
                    if source == file_source:
                        labels.append(np.uint32(label))
                        break
            labels = np.array(labels)
            onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
            onehot[np.arange(labels.size), labels] = 1.0
            tfr.add_labels(onehot[order])

#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=' ') # The input directory containing subdirectories of images. Each subdirectory represents a data source, either from the real dataset or generated by a GAN
    parser.add_argument('--out_dir', type=str, default=' ') #  The output directory containing the prepared data format that enables efficient streaming
    parser.add_argument('--resolution', type=int, default=128) # The resolution to which images are resized
    parser.add_argument('--shuffle', type=int, default=1) # Shuffle the order of images when streaming?
    parser.add_argument('--export_labels', type=int, default=1) # Export image source labels?
    parser.add_argument('--percentage_samples', type=int, default=100) # The percentage of images used for data preparation
    args = parser.parse_args()
    data_preparation(args.in_dir, args.out_dir, args.resolution, args.shuffle, args.export_labels, args.percentage_samples)