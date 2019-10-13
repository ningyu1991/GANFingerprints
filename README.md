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
To train GANs and our classifiers, we consider two real-world datasets:
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) aligned and cropped face dataset. We crop each image centered at (x,y) = (89,121) with size 128x128 before training.
- [LSUN](https://github.com/fyu/lsun) bedroom scene dataset. We select the first 200k images, center-crop them to square size according to the shorter side length, and resize them to 128x128 before training.
  
## GAN Models
For each dataset, we pre-train four GAN sources:
- [ProGAN](https://github.com/tkarras/progressive_growing_of_gans)
  - Follow their "Preparing datasets for training" Section for dataset preparation. Use the `create_from_images` option in `./ProGAN/dataset_tool.py`. The prepared data enables efficient streaming.
  - After data preparation, run, e.g.,
    ```
    cd ProGAN/
    python3 run.py \
    --app train \
    --training_data_dir datasets/celeba_align_png_cropped_seed_v0/ \
    --out_model_dir models/celeba_align_png_cropped_seed_v0/ \
    --training_seed 0
    ```
    where
    - `training_data_dir`: The prepared training dataset directory that can be efficiently called by the code.
    - `out_model_dir`: The output directory containing trained models, training configureations, training log, and training snapshots.
    - `training_seed`: The random seed that differentiates training instances.
  - Or download our pre-trained models [here](https://drive.google.com/drive/folders/1E4Bm8xshBTDPBU3Nh8x6ASFduLZZmtVI?usp=sharing) and put them at `./ProGAN/models/`. The models named with `_seed_v%d` are only different in random seeds from each other.
  - With pre-trained model, generate images by running, e.g.,
    ```
    cd ProGAN/
    python3 run.py \
    --app gen \
    --model_path models/celeba_align_png_cropped.pkl \
    --out_image_dir gen/celeba_align_png_cropped/ \
    --num_pngs 10000 \
    --gen_seed 0
    ```
    where
    - `model_path`: The well-trained GAN model in the format of .pkl.
    - `out_image_dir`: The outpupt directory containing generated images.
    - `num_pngs`: The number of generated images in the format of .png.
    - `gen_seed`: The random seed that differentiates generation instances.
- [SNGAN](https://github.com/pfnet-research/sngan_projection)
  - Their code requires Python 2.7 instead of Python 3.6, and additionally requires: chainer, cython, cupy, pyyaml.
  
