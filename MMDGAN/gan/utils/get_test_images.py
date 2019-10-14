import tensorflow as tf
import numpy as np
import os
os.chdir(os.path.join(os.getcwd(), '..', '..'))
import core.pipeline
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='lsun', help='dataset to sample from')
    parser.add_argument('--data-path', default='~/', help='path to read raw images from')
    parser.add_argument('--save-path', default=None, help='path to save numpy array of images')
    parser.add_argument('-N', type=int, default=25000, help='number of samples to save')
    parser.add_argument('--output-size', type=int, default=64, help='size of the sampled pictures')
    parser.add_argument('--channels', type=int, default=3, help='number of channels in sampled pictures')
    args = parser.parse_args()
    
    if args.save_path is None:
        args.save_path = args.data_path

    Pipeline_class = pipeline.get_pipeline(args.dataset)
    with tf.Session() as sess:
        pipe = Pipeline(args.output_size, args.channels, 1000, args.save_path)
        ims = pipe.connect()
        sampled = []
        while len(sampled) < args.N/1000.:
            sampled.append(sess.run(ims))
            print(len(sampled))
        sampled = np.concatenate(sampled, axis=0)
        print(sampled.shape)
        sampled = sampled[:args.N]
        
        path = os.path.join(args.save_path, '%s-$d-test.npy')
        np.save(sampled, path)
        print('%d %dx%d %s images saved in %s.' % (args.N, args.output_size, args.output_size, args.dataset, path))
        
