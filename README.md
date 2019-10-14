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
  - **Data preparation**. Follow their "Preparing datasets for training" Section for dataset preparation. Use the `create_from_images` option in `./ProGAN/dataset_tool.py`. The prepared data enables efficient streaming.
  - **Training**. Run, e.g.,
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
    - `out_model_dir`: The output directory containing trained models, training configureation, training log, and training snapshots.
    - `training_seed`: The random seed that differentiates training instances.
  - **Pre-trained models**. Download our pre-trained models [here](https://drive.google.com/drive/folders/1E4Bm8xshBTDPBU3Nh8x6ASFduLZZmtVI?usp=sharing) and put them at `./ProGAN/models/`. The models named with `_seed_v%d` are only different in random seeds from each other.
  - **Generation**. With pre-trained model, generate images by running, e.g.,
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
    - `model_path`: The pre-trained GAN model.
    - `out_image_dir`: The outpupt directory containing generated images.
    - `num_pngs`: The number of generated images.
    - `gen_seed`: The random seed that differentiates generation instances.
- [SNGAN](https://github.com/pfnet-research/sngan_projection)
  - **Additional Python dependencies**: chainer, cython, cupy, pyyaml.
  - **Data preparation**. Run
    ```
    cd SNGAN/datasets/
    python3 celeba.py ../../celeba_align_png_cropped/
    ```
    Or
    ```
    cd SNGAN/datasets/
    python3 lsun_bedroom_200k.py ../../lsun_bedroom_train_200k_png/
    ```
    where `../../img_align_celeba_png/` and `../../lsun_bedroom_train_200k/` are the training dataset directories containing 128x128 png images.
  - **Training**. Run, e.g.,
    ```
    cd SNGAN/
    python3 train.py \
    --config_path configs/sn_projection_celeba.yml \
    --data_dir ../celeba_align_png_cropped/ \
    --results_dir models/celeba_align_png_cropped \
    --gpu 0 \
    --loaderjob 32
    ```
    where
    - `config_path`: The training configuration file. The yml file for LSUN is at `configs/sn_projection_lsun_bedroom_200k.yml`.
    - `data_dir`: The training dataset directory containing 128x128 png images.
    - `results_dir`: The output directory containing trained models, training configureations, training log, and training snapshots.
    - `gpu`: The GPU device ID to use.
    - `loaderjob`: The number of parallel data loading processes.
  - **Pre-trained models**. Download our pre-trained models [here](https://drive.google.com/drive/folders/1q9o8qToZ44WifVBRdu9LM6-DHjl3oi9f?usp=sharing) and put them at `./SNGAN/models/`.
  - **Generation**. Run, e.g.,
    ```
    cd SNGAN/
    python3 evaluations/gen_images.py \
    --config_path configs/sn_projection_celeba.yml \
    --snapshot models/celeba_align_png_cropped.npz \
    --results_dir gen/celeba_align_png_cropped \
    --num_pngs 10000 \
    --seed 0 \
    --gpu 0
    ```
    where
    - `config_path`: The training configuration file. The yml file for LSUN is at `configs/sn_projection_lsun_bedroom_200k.yml`.
    - `snapshot`: The pre-trained GAN model.
    - `results_dir`: The outpupt directory containing generated images.
    - `num_pngs`: The number of generated images in the format of .png.
    - `seed`: The random seed that differentiates generation instances.
    - `gpu`: The GPU device ID to use.
