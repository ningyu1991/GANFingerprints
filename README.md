# GANFingerprints

<img src='fig/teaser.png' width=800>

- Official Tensorflow implementation for our [ICCV'19 paper](https://arxiv.org/pdf/1811.08180.pdf) on GAN fingerprints definition, learning, visualization, and analysis.
- Contact: Ning Yu (ningyu AT mpi-inf DOT mpg DOT de)

## Abstract
Recent advances in Generative Adversarial Networks (GANs) have shown increasing success in generating photorealistic images. But they also raise challenges to visual forensics and model attribution. We present the first study of learning GAN fingerprints towards image attribution and using them to classify an image as real or GAN-generated. For GAN-generated images, we further identify their sources. Our experiments show that:
- GANs carry distinct model fingerprints and leave stable fingerprints in their generated images, which support image attribution;
- Even minor differences in GAN training can result in different fingerprints, which enables fine-grained model authentication;
- Fingerprints persist across different image frequencies and patches and are not biased by GAN artifacts;
- Fingerprint finetuning is effective in immunizing against five types of adversarial image perturbations;
- Comparisons also show our learned fingerprints consistently outperform several baselines in a variety of setups.

## Prerequisites
- Linux
- NVIDIA GPU + CUDA + CuDNN
- Python 3.6
- tensorflow-gpu
- Other Python dependencies: numpy, scipy, moviepy, Pillow, skimage, lmdb, opencv-python, cryptography, h5py, six

## Datasets
- To train GANs and our classifiers, we consider two real-world datasets:
  - [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) aligned and cropped face dataset. We crop each image centered at (89,121) with size 128x128 before training.
  - [LSUN](https://github.com/fyu/lsun) bedroom scene dataset. We select the first 200k images, crop them to square size according to the shorter side length, and resize them to 128x128 before training.
- For each dataset, we pre-train four GAN sources:
  - [ProGAN](https://github.com/tkarras/progressive_growing_of_gans). Our pre-trained models are saved as `network-final.pkl` in each subdirectory of `./ProGAN/`. The training configurations are saved as `config.txt` in each subdirectory.
