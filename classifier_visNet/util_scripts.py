# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import os.path
import numpy as np
import PIL.Image
import skimage
import skimage.io
import skimage.transform

import misc

def classify(model_path, testing_data_path, out_fingerprint_dir):

    labels_1 = ['CelebA_real_data', 'ProGAN_generated_data', 'SNGAN_generated_data', 'CramerGAN_generated_data', 'MMDGAN_generated_data']
    labels_2 = ['CelebA_real_data', 'ProGAN_seed_0_generated_data ', 'ProGAN_seed_1_generated_data', 'ProGAN_seed_2_generated_data', 'ProGAN_seed_3_generated_data', 'ProGAN_seed_4_generated_data', 'ProGAN_seed_5_generated_data', 'ProGAN_seed_6_generated_data', 'ProGAN_seed_7_generated_data', 'ProGAN_seed_8_generated_data', 'ProGAN_seed_9_generated_data']

    print('Loading network...')
    EG, D_rec, EGs = misc.load_network_pkl(model_path)
    est_fingerprints = np.transpose(EGs.vars['Conv_fingerprints/weight'].eval(), axes=[3,2,0,1])
    est_fingerprints_min = np.amin(est_fingerprints)
    est_fingerprints_max = np.amax(est_fingerprints)
    if est_fingerprints.shape[0] == len(labels_1):
        labels = list(labels_1)
    elif est_fingerprints.shape[0] == len(labels_2):
        labels = list(labels_2)
    for idx, label in enumerate(labels):
        path = '%s/model_fingerprint_%s.png' % (out_fingerprint_dir, label)
        skimage.io.imsave(path, misc.adjust_dynamic_range(np.transpose(est_fingerprints[idx,:,:,:], [1,2,0]), [est_fingerprints_min,est_fingerprints_max], [0,255]).astype(np.uint8))

    if testing_data_path.endswith('.png') or testing_data_path.endswith('.jpg'):
        im = np.array(PIL.Image.open(testing_data_path)).astype(np.float32) / 255.0
        if len(im.shape) < 3:
            im = np.dstack([im, im, im])
        if im.shape[2] == 4:
            im = im[:,:,:3]
        if im.shape[0] != 128:
            im = skimage.transform.resize(im, (128, 128))
        im = np.transpose(misc.adjust_dynamic_range(im, [0,1], [-1,1]), axes=[2,0,1])
        im = np.reshape(im, [1]+list(im.shape))
        rec, fingerprint, logits = EGs.run(im, minibatch_size=1, num_gpus=1, out_dtype=np.float32)
        idx = np.argmax(np.squeeze(logits))
        print('The input image is predicted as being sampled from %s' % labels[idx])
        path = '%s/image_fingerprint.png' % out_fingerprint_dir
        skimage.io.imsave(path, misc.adjust_dynamic_range(np.transpose(fingerprint[0,:,:,:], [1,2,0]), [-1.0,1.0], [0,255]).astype(np.uint8))
        for idx, label in enumerate(labels):
            masked_fingerprint = fingerprint[0,:,:,:] * est_fingerprints[idx,:,:,:]
            masked_fingerprint_flag = np.amin(masked_fingerprint, axis=0, keepdims=True)
            masked_fingerprint_flag = np.tile(masked_fingerprint_flag, [3,1,1])
            masked_fingerprint[masked_fingerprint_flag<0.0] = 0.0 # for cleaner visualization
            path = '%s/image_fingerprint_responding_to_model_fingerprint_%s.png' % (out_fingerprint_dir, label)
            skimage.io.imsave(path, misc.adjust_dynamic_range(np.transpose(masked_fingerprint, [1,2,0]), [-5.0,5.0], [0,255]).astype(np.uint8))
    
    elif os.path.isdir(testing_data_path):
        count_dict = None
        name_list = sorted(os.listdir(testing_data_path))
        length = len(name_list)
        for (count0, name) in enumerate(name_list):
            im = np.array(PIL.Image.open('%s/%s' % (testing_data_path, name))).astype(np.float32) / 255.0
            if len(im.shape) < 3:
                im = np.dstack([im, im, im])
            if im.shape[2] == 4:
                im = im[:,:,:3]
            if im.shape[0] != 128:
                im = skimage.transform.resize(im, (128, 128))
            im = np.transpose(misc.adjust_dynamic_range(im, [0,1], [-1,1]), axes=[2,0,1])
            im = np.reshape(im, [1]+list(im.shape))
            rec, fingerprint, logits = EGs.run(im, minibatch_size=1, num_gpus=1, out_dtype=np.float32)
            idx = np.argmax(np.squeeze(logits))
            if count_dict is None:
                count_dict = {}
                for label in labels:
                    count_dict[label] = 0
            count_dict[labels[idx]] += 1
            print('Classifying %d/%d images: %s: predicted as being sampled from %s' % (count0, length, name, labels[idx]))
            path = '%s/%s_image_fingerprint.png' % (out_fingerprint_dir, name[:-4])
            skimage.io.imsave(path, misc.adjust_dynamic_range(np.transpose(fingerprint[0,:,:,:], [1,2,0]), [-1.0,1.0], [0,255]).astype(np.uint8))
            for idx, label in enumerate(labels):
                masked_fingerprint = fingerprint[0,:,:,:] * est_fingerprints[idx,:,:,:]
                masked_fingerprint_flag = np.amin(masked_fingerprint, axis=0, keepdims=True)
                masked_fingerprint_flag = np.tile(masked_fingerprint_flag, [3,1,1])
                masked_fingerprint[masked_fingerprint_flag<0.0] = 0.0 # for cleaner visualization
                path = '%s/%s_image_fingerprint_responding_to_model_fingerprint_%s.png' % (out_fingerprint_dir, name[:-4], label)
                skimage.io.imsave(path, misc.adjust_dynamic_range(np.transpose(masked_fingerprint, [1,2,0]), [-5.0,5.0], [0,255]).astype(np.uint8))
        for label in labels:
            print('The percentage of images sampled from %s is %d/%d = %.2f%%' % (label, count_dict[label], length, float(count_dict[label])/float(length)*100.0))