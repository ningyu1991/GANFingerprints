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
desc        = 'pgan'                                        # Description string included in result subdir name.
random_seed = 1000                                          # Global random seed.
train       = EasyDict(func='run.train_classifier') # Options for main training func.
C_im       	= EasyDict(func='networks.C_patch', fmap_base=1024, fmap_max=512, latent_res=-1, mode=mode, switching_res=switching_res)             # Options for classifier network in the image domain.
C_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for classifier optimizer.
C_loss      = EasyDict(func='loss.C_classification')        # Options for classifier loss.
sched       = EasyDict()                                    # Options for train.TrainingSchedule.
grid        = EasyDict(size='1080p', layout='random')       # Options for train.setup_snapshot_image_grid().

# Dataset (choose one).
desc += '-%s' % training_data; training_set = EasyDict(tfrecord_dir=training_data); validation_set = EasyDict(tfrecord_dir=validation_data); train.lr_mirror_augment = True; train.ud_mirror_augment = False; sched.lod_initial_resolution = 128; sched.lod_training_kimg = 100000; sched.lod_transition_kimg = 100000; train.total_kimg = 100000

# Conditioning & snapshot options.
desc += '-labels'; training_set.max_label_size = 'full'; validation_set.max_label_size = 'full' # conditioned on full label

# Config presets (choose one). Note: the official settings are optimal. It is not the larger batch size the better.
desc += '-preset-v2-1gpu'; num_gpus = 1; sched.minibatch_base = 32; sched.lrate_dict = {1024: 0.0015}

# Numerical precision (choose one).
desc += '-fp32'; sched.max_minibatch_per_gpu = {256: 16, 512: 8, 1024: 4}





#num_gpus = 1; env.CUDA_VISIBLE_DEVICES = '0'
#out_dir = '%s/val_classification_samples' % result_dir
#out_dir = '%s/val_classification_samples_more_realistic_perceptual_1NN' % result_dir
#out_dir = '%s/val_classification_samples_superresolution_perturbation_attack_val_128' % result_dir
#train = EasyDict(func='util_scripts.accuracy_confusion_matrix_saving_samples', out_dir=out_dir)



'''
num_gpus = 1; env.CUDA_VISIBLE_DEVICES = '2'
out_dir = '%s/val_sample_modification' % result_dir
train = EasyDict(func='util_scripts.modify_images', out_dir=out_dir)
'''

'''
num_gpus = 1; env.CUDA_VISIBLE_DEVICES = '1'
in_dir = '../MultiGAN_dataset/val/CelebA'; out_dir = '../MultiGAN_dataset/val/CelebA_more_realistic'
train = EasyDict(func='util_scripts.more_realistic_picker', in_dir=in_dir, out_dir=out_dir, ratio=0.1)
'''

'''
num_gpus = 1; env.CUDA_VISIBLE_DEVICES = '0'
base_path = '../MultiGAN_dataset/val'
if validation_data == 'CelebA_ProGAN_SNGAN_CRAMER_MMD_val_128':
	class_list = ['CelebA', 'ProGAN', 'SNGAN', 'CRAMER', 'MMD']
elif validation_data == 'CelebA_ProGAN_overlap_subset_shifts_100k_x11_val_128':
	class_list = ['CelebA', 'ProGAN_overlap_subset_shifts_base1_100k', 'ProGAN_overlap_subset_shifts_1_100k', 'ProGAN_overlap_subset_shifts_10_100k', 'ProGAN_overlap_subset_shifts_100_100k', 'ProGAN_overlap_subset_shifts_1000_100k', 'ProGAN_overlap_subset_shifts_10000_100k', 'ProGAN_overlap_subset_shifts_20000_100k', 'ProGAN_overlap_subset_shifts_40000_100k', 'ProGAN_overlap_subset_shifts_60000_100k', 'ProGAN_overlap_subset_shifts_80000_100k', 'ProGAN_overlap_subset_shifts_100000_100k']
elif validation_data == 'CelebA_ProGAN_seed_x10_val_128':
	class_list = ['CelebA', 'ProGAN_seed_v0', 'ProGAN_seed_v1', 'ProGAN_seed_v2', 'ProGAN_seed_v3', 'ProGAN_seed_v4', 'ProGAN_seed_v5', 'ProGAN_seed_v6', 'ProGAN_seed_v7', 'ProGAN_seed_v8', 'ProGAN_seed_v9']
elif validation_data == 'CelebA_ProGAN_seed_x10_more_realistic_perceptual_1NN_val_128':
	class_list = ['CelebA_more_realistic', 'ProGAN_seed_v0_more_realistic_perceptual_1NN', 'ProGAN_seed_v1_more_realistic_perceptual_1NN', 'ProGAN_seed_v2_more_realistic_perceptual_1NN', 'ProGAN_seed_v3_more_realistic_perceptual_1NN', 'ProGAN_seed_v4_more_realistic_perceptual_1NN', 'ProGAN_seed_v5_more_realistic_perceptual_1NN', 'ProGAN_seed_v6_more_realistic_perceptual_1NN', 'ProGAN_seed_v7_more_realistic_perceptual_1NN', 'ProGAN_seed_v8_more_realistic_perceptual_1NN', 'ProGAN_seed_v9_more_realistic_perceptual_1NN']
elif validation_data == 'LSUN_ProGAN_SNGAN_CRAMER_MMD_lsun_bedroom_200k_val_128':
	class_list = ['LSUN_bedroom_200k', 'ProGAN_lsun_bedroom_200k', 'SNGAN_lsun_bedroom_200k', 'CRAMER_lsun_bedroom_200k', 'MMD_lsun_bedroom_200k']
elif validation_data == 'LSUN_ProGAN_overlap_subset_shifts_100k_x11_lsun_bedroom_200k_val_128':
    class_list = ['LSUN_bedroom_200k', 'ProGAN_lsun_bedroom_200k_overlap_subset_shifts_0_100k', 'ProGAN_lsun_bedroom_200k_overlap_subset_shifts_1_100k', 'ProGAN_lsun_bedroom_200k_overlap_subset_shifts_10_100k', 'ProGAN_lsun_bedroom_200k_overlap_subset_shifts_100_100k', 'ProGAN_lsun_bedroom_200k_overlap_subset_shifts_1000_100k', 'ProGAN_lsun_bedroom_200k_overlap_subset_shifts_10000_100k', 'ProGAN_lsun_bedroom_200k_overlap_subset_shifts_20000_100k', 'ProGAN_lsun_bedroom_200k_overlap_subset_shifts_40000_100k', 'ProGAN_lsun_bedroom_200k_overlap_subset_shifts_60000_100k', 'ProGAN_lsun_bedroom_200k_overlap_subset_shifts_80000_100k', 'ProGAN_lsun_bedroom_200k_overlap_subset_shifts_100000_100k']
elif validation_data == 'LSUN_ProGAN_seed_x10_lsun_bedroom_200k_val_128':
    class_list = ['LSUN_bedroom_200k', 'ProGAN_lsun_bedroom_200k_seed_v0', 'ProGAN_lsun_bedroom_200k_seed_v1', 'ProGAN_lsun_bedroom_200k_seed_v2', 'ProGAN_lsun_bedroom_200k_seed_v3', 'ProGAN_lsun_bedroom_200k_seed_v4', 'ProGAN_lsun_bedroom_200k_seed_v5', 'ProGAN_lsun_bedroom_200k_seed_v6', 'ProGAN_lsun_bedroom_200k_seed_v7', 'ProGAN_lsun_bedroom_200k_seed_v8', 'ProGAN_lsun_bedroom_200k_seed_v9']
elif validation_data == 'LSUN_ProGAN_seed_x10_lsun_bedroom_200k_more_realistic_perceptual_1NN_val_128':
    class_list = ['LSUN_bedroom_200k_more_realistic', 'ProGAN_lsun_bedroom_200k_seed_v0_more_realistic_perceptual_1NN', 'ProGAN_lsun_bedroom_200k_seed_v1_more_realistic_perceptual_1NN', 'ProGAN_lsun_bedroom_200k_seed_v2_more_realistic_perceptual_1NN', 'ProGAN_lsun_bedroom_200k_seed_v3_more_realistic_perceptual_1NN', 'ProGAN_lsun_bedroom_200k_seed_v4_more_realistic_perceptual_1NN', 'ProGAN_lsun_bedroom_200k_seed_v5_more_realistic_perceptual_1NN', 'ProGAN_lsun_bedroom_200k_seed_v6_more_realistic_perceptual_1NN', 'ProGAN_lsun_bedroom_200k_seed_v7_more_realistic_perceptual_1NN', 'ProGAN_lsun_bedroom_200k_seed_v8_more_realistic_perceptual_1NN', 'ProGAN_lsun_bedroom_200k_seed_v9_more_realistic_perceptual_1NN']
feature_type = 'inception'
#feature_type = 'fingerprint'
train = EasyDict(func='util_scripts.FID_matrix', base_path=base_path, class_list=class_list, feature_type=feature_type)
'''


#train = EasyDict(func='util_scripts.generate_fake_images', run_id=23, num_pngs=1000); num_gpus = 1; desc = 'fake-images-' + str(train.run_id)

'''
num_gpus = 1; env.CUDA_VISIBLE_DEVICES = '0'
out_dir = '%s/val_tsne_for_teaser' % result_dir
if validation_data == 'CelebA_ProGAN_SNGAN_CRAMER_MMD_val_128':
	#class_list = ['CelebA', 'ProGAN', 'SNGAN', 'CramerGAN', 'MMDGAN']
	class_list = ['Real', 'GAN: Arch X, Data Y, Seed Z', 'GAN: Arch A, Data A, Seed A', 'GAN: Arch A, Data B, Seed B', 'GAN: Arch A, Data B, Seed C', 'GAN: Arch C, Data C, Seed C']
elif validation_data == 'CelebA_ProGAN_overlap_subset_shifts_100k_x11_val_128':
	class_list = ['CelebA', 'ProGAN_subset_diff_0', 'ProGAN_subset_diff_1', 'ProGAN_subset_diff_10', 'ProGAN_subset_diff_100', 'ProGAN_subset_diff_1k', 'ProGAN_subset_diff_10k', 'ProGAN_subset_diff_20k', 'ProGAN_subset_diff_40k', 'ProGAN_subset_diff_60k', 'ProGAN_subset_diff_80k', 'ProGAN_subset_diff_100k']
else:
	class_list = ['CelebA', 'ProGAN_seed_v0', 'ProGAN_seed_v1', 'ProGAN_seed_v2', 'ProGAN_seed_v3', 'ProGAN_seed_v4', 'ProGAN_seed_v5', 'ProGAN_seed_v6', 'ProGAN_seed_v7', 'ProGAN_seed_v8', 'ProGAN_seed_v9']
#feature_type = 'inception'
feature_type = 'fingerprint'
train = EasyDict(func='util_scripts.val_tsne', out_dir=out_dir, class_list=class_list, feature_type=feature_type, num_images=1000)
'''

'''
dataset.tfrecord_dir = 'flickr_earth_textures_new_test_aug_crop_128_with_labels'
#train = EasyDict(func='util_scripts.generate_spatial_interpolation_images', run_id=0, snapshot=None, changes=10, scale=8, repeats=False); num_gpus = 1; desc = 'interpolation-image-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_spatial_interpolation_image', scale_h=8, scale_w=8, repeats=False); num_gpus = 1; desc = 'interpolation-image'
#train = EasyDict(func='util_scripts.generate_temporal_interpolation_video', grid_size=[15,8], duration_sec=4.0); env.CUDA_VISIBLE_DEVICES = '0'; num_gpus = 1; desc = 'interpolation-video'
#train = EasyDict(func='util_scripts.generate_spatiotemporal_interpolation_video', run_id=0, snapshot=None, grid_size=[4,2], scale=4, duration_sec=5.0, repeats=True); num_gpus = 1; desc = 'interpolation-video-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_spatiotemporal_interpolation_video', run_id=0, snapshot=None, grid_size=[2,1], scale=8, duration_sec=5.0, repeats=False); num_gpus = 1; desc = 'interpolation-video-' + str(train.run_id)
train = EasyDict(func='util_scripts.generate_spatiotemporal_interpolation_plus_reconstructed_corners_video', changes=1, scale_h=8, scale_w=8, duration_sec=2.0, repeats=False); env.CUDA_VISIBLE_DEVICES = '0'; num_gpus = 1; desc = 'interpolation-video'
#train = EasyDict(func='util_scripts.generate_anchor_movement_interpolation_video', run_id=0, snapshot=None, texture_id=76, scale=8, duration_sec=20.0, mode='only_one_brushing', stroke_radius_div=24.0); num_gpus = 1; desc = 'interpolation-video-' + str(train.run_id)
'''






#data_path = '../flickr_datasets/earth_textures_new/images/test_resize512/'
#train = EasyDict(func='util_scripts.generate_spatiotemporal_interpolation_plus_rotated_reconstructed_corners_video', data_path=data_path, run_id=0, snapshot=None, changes=10, scale=3, duration_sec=50.0); num_gpus = 1; desc = 'interpolation-video-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_rotating_spatiotemporal_interpolation_plus_reconstructed_corners_video', data_path=data_path, run_id=0, snapshot=None, texture_id=2, delta_rotation=5, scale=8, duration_sec=50.0); num_gpus = 1; desc = 'interpolation-video-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_spatiotemporal_interpolation_plus_resized_reconstructed_corners_video', data_path=data_path, run_id=0, snapshot=None, changes=10, scale=8, duration_sec=50.0); num_gpus = 1; desc = 'interpolation-video-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_resizing_spatiotemporal_interpolation_plus_reconstructed_corners_video', data_path=data_path, run_id=0, snapshot=None, texture_id=2, delta_log_scale=-0.02, scale=8, duration_sec=50.0); num_gpus = 1; desc = 'interpolation-video-' + str(train.run_id)







painting_path = '../painting/beach/painting.png'
mask_path = '../painting/beach/mask.png'
#train = EasyDict(func='util_scripts.generate_painting_brush_images', painting_path=painting_path, mask_path=mask_path, run_id=0, snapshot=None, texture_id=-2); num_gpus = 1; desc = 'interpolation-image-' + str(train.run_id)

#train = EasyDict(func='util_scripts.count_generated_labels', run_id=0, snapshot=8999, num_samples=10000, grid_size=[15,8]); num_gpus = 1; desc = 'generated_label_counts-' + str(train.run_id)

#train = EasyDict(func='util_scripts.reconstruct_BFGS_images', run_id=0, snapshot=8999, grid_size=[15,8]); num_gpus = 1; desc = 'BFGS-reconstruction-images-' + str(train.run_id)

#train = EasyDict(func='util_scripts.generate_training_video', run_id=0, duration_sec=30.0); num_gpus = 1; desc = 'training-video-' + str(train.run_id)

#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=0, snapshot=27784, log='metric-swd-16k.txt', metrics=['swd'], num_images=16384, real_passes=2); num_gpus = 1; desc = log.split('.')[0] + '-' + str(train.run_id) + '-' + str(train.snapshot)
#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-fid-10k.txt', metrics=['fid'], num_images=10000, real_passes=1); num_gpus = 1; desc = log.split('.')[0] + '-' + str(train.run_id)
#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-fid-50k.txt', metrics=['fid'], num_images=50000, real_passes=1); num_gpus = 1; desc = log.split('.')[0] + '-' + str(train.run_id)
#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-is-50k.txt', metrics=['is'], num_images=50000, real_passes=1); num_gpus = 1; desc = log.split('.')[0] + '-' + str(train.run_id)
#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-msssim-20k.txt', metrics=['msssim'], num_images=20000, real_passes=1); num_gpus = 1; desc = log.split('.')[0] + '-' + str(train.run_id)








'''
result_dir = 'results/disabled_zg_hard_zl_disabled_zp_latent_512_scale_3_gen_G_0.00_rec_G_1.00_pixel_50.00_perceptual_0.00_gram_0.0005_autocorrelation_50.00_class_1.00_latent_0.00_KL_0.00_interp_G_0.00_flickr_earth_textures_new_train_aug_crop_128_with_labels'; zg_variational = 'disabled'; zl_variational = 'hard'; zp_variational = 'disabled'
dataset.tfrecord_dir = 'flickr_earth_textures_new_test_aug_crop_128_with_labels'; log = 'test_reconstruction.txt'; num_images = 10094
env.CUDA_VISIBLE_DEVICES = '0'; num_gpus = 1
train = EasyDict(func='util_scripts.reconstruction_evaluation', run_id=2, snapshot=None, log=log, num_images=num_images, minibatch_size=64*num_gpus); desc = log.split('.')[0] + '-' + str(train.run_id)
'''
'''
result_dir = 'results/latent_res_EG_32_latent_res_DC_-1_latent_channels_128_gen_G_0.00_rec_G_1.00_pixel_200.00_perceptual_0.00_gram_0.0020_autocorrelation_0.00_class_0.00_latent_0.00_KL_10.00_interp_G_1.00_flickr_earth_textures_new_train_aug_crop_128_with_labels'
#dataset.tfrecord_dir = 'flickr_earth_textures_new_train_aug_crop_128_with_labels'; log = 'train_synthesis.txt'; in_dir = '../flickr_datasets/earth_textures_new/images/train_resize512_aug_crop_128'; out_dir = 'output_horizontal_texture_synthesis_via_interpolation_earth_textures_new_train_resize512_aug_crop_128'; num_images = 10094
dataset.tfrecord_dir = 'flickr_earth_textures_new_test_aug_crop_128_with_labels'; log = 'test_synthesis.txt'; in_dir = '../flickr_datasets/earth_textures_new/images/test_resize512_aug_crop_128'; out_dir = '%s/output_single_texture_synthesis_via_interpolation_earth_textures_new_test_resize512_aug_crop_128' % result_dir; num_images = None
train = EasyDict(func='util_scripts.dataset_synthesis', run_id=0, snapshot=None, log=log, in_dir=in_dir, out_dir=out_dir, num_images=num_images, minibatch_size=64*num_gpus, scale_h=1, scale_w=8)
'''




'''
env.CUDA_VISIBLE_DEVICES = '0'; num_gpus = 1
#result_dir = 'results/swap_4_latentResEG_32_latentResDC_-1_latentChannels_128_genG_0.00_recG_1.00_pixel_200.00_perceptual_0.00_gram_0.0020_aCorr_0.00_classIm_0.00_latent_0.00_KL_0.00_interpG_1.00_adobe_stock_animal_textures_new_train_aug_crop_128_with_labels'
#dataset.tfrecord_dir = 'flickr_earth_textures_new_train_aug_crop_128_with_labels'; in_dir = '../flickr_datasets/earth_textures_new/images/train_resize512_aug_crop_128'; out_dir = 'output_horizontal_texture_interpolation_via_interpolation_earth_textures_new_train_resize512_aug_crop_128'; num_images = 10094
dataset.tfrecord_dir = 'flickr_earth_textures_new_test_aug_crop_128_with_labels'; in_dir = '../flickr_datasets/earth_textures_new/images/test_resize512_aug_crop_128'; out_dir = '%s/[zg_zl]global_swap_4_genG_blendG_output_horizontal_texture_interpolation_via_interpolation_earth_textures_new_test_resize512_aug_crop_128' % (result_dir); num_images = None
#dataset.tfrecord_dir = 'adobe_stock_animal_textures_new_test_aug_crop_128_with_labels'; in_dir = '../adobe_stock_datasets/animal_textures_new/images/test_resize512_aug_crop_128'; out_dir = '%s/global_swap_4_output_horizontal_texture_interpolation_via_interpolation_animal_textures_new_test_resize512_aug_crop_128' % (result_dir); num_images = None
#dataset.tfrecord_dir = 'adobe_stock_plant_textures_new_test_aug_crop_128_with_labels'; in_dir = '../adobe_stock_datasets/plant_textures_new/images/test_resize512_aug_crop_128'; out_dir = '%s/perm_1_output_horizontal_texture_interpolation_via_interpolation_plant_textures_new_test_resize512_aug_crop_128' % (result_dir); num_images = None
train = EasyDict(func='util_scripts.dataset_interpolation', in_dir=in_dir, out_dir=out_dir, num_images=num_images, minibatch_size=32*num_gpus, scale_h=1, scale_w=8)
'''






'''
#env.CUDA_VISIBLE_DEVICES = '0'; num_gpus = 1
dataset.tfrecord_dir = 'flickr_earth_textures_new_test_aug_crop_128_with_labels'; src_dir = '../flickr_datasets/earth_textures_new/images/test_resize512_aug_crop_128'
#dataset.tfrecord_dir = 'adobe_stock_animal_textures_new_test_aug_crop_128_with_labels'; src_dir = '../adobe_stock_datasets/animal_textures_new/images/test_resize512_aug_crop_128'
#interp_dir = 'results/swap_8_latentResEG_32_latentResDC_-1_latentChannels_128_genG_0.00_recG_1.00_pixel_200.00_perceptual_0.00_gram_0.0020_aCorr_0.00_classIm_0.00_latent_0.00_KL_0.00_interpG_1.00_flickr_earth_textures_new_train_aug_crop_128_with_labels/global_swap_8_genG_output_horizontal_texture_interpolation_via_interpolation_earth_textures_new_test_resize512_aug_crop_128'; num_images = None
#interp_dir = 'results/swap_4_latentResEG_32_latentResDC_-1_latentChannels_128_genG_0.00_recG_1.00_pixel_200.00_perceptual_0.00_gram_0.0020_aCorr_0.00_classIm_0.00_latent_0.00_KL_0.00_interpG_1.00_adobe_stock_animal_textures_new_train_aug_crop_128_with_labels/global_swap_4_output_horizontal_texture_interpolation_via_interpolation_animal_textures_new_test_resize512_aug_crop_128'; num_images = None
interp_dir = '%s/[zg_zl]global_swap_4_genG_blendG_output_horizontal_texture_interpolation_via_interpolation_earth_textures_new_test_resize512_aug_crop_128' % result_dir; num_images = None
#interp_dir = '../progressive_growing_of_zg_zl_splittable_only_vae_res_interp_gans-tensorflow/results/perm_1_latentResEG_32_latentResDC_-1_latentChannels_128_genG_0.00_recG_1.00_pixel_200.00_perceptual_0.00_gram_0.0020_aCorr_0.00_classIm_0.00_classLatent_1.00_latent_0.00_KL_10.00_interpG_1.00_flickr_earth_textures_new_train_aug_crop_128_with_labels/perm_1_output_horizontal_texture_interpolation_via_interpolation_earth_textures_new_test_resize512_aug_crop_128'; num_images = None
#interp_dir = '../progressive_growing_of_zg_zl_splittable_only_vae_res_interp_gans-tensorflow/results/swap_1_latentResEG_32_latentResDC_-1_latentChannels_128_genG_0.00_recG_1.00_pixel_200.00_perceptual_0.00_gram_0.0020_aCorr_0.00_classIm_0.00_classLatent_1.00_latent_0.00_KL_10.00_interpG_1.00_flickr_earth_textures_new_train_aug_crop_128_with_labels/swap_1_output_horizontal_texture_interpolation_via_interpolation_earth_textures_new_test_resize512_aug_crop_128'; num_images = None
#interp_dir = '../progressive_growing_of_zg_zl_splittable_only_vae_res_interp_gans-tensorflow/results/swap_4_latentResEG_32_latentResDC_-1_latentChannels_128_genG_0.00_recG_1.00_pixel_200.00_perceptual_0.00_gram_0.0020_aCorr_0.00_classIm_0.00_classLatent_1.00_latent_0.00_KL_10.00_interpG_1.00_flickr_earth_textures_new_train_aug_crop_128_with_labels/swap_4_output_horizontal_texture_interpolation_via_interpolation_earth_textures_new_test_resize512_aug_crop_128'; num_images = None
#interp_dir = '../../image_melding/Results/output_horizontal_texture_interpolation_via_interpolation_earth_textures_new_test_resize512_aug_crop_128'; num_images = None
#interp_dir = '../AdaIN-style-master/output_horizontal_texture_interpolation_via_stylizing_noise_earth_textures_new_test_resize512_aug_crop_128'; num_images = None
#interp_dir = '../WCT_code_for_chen/output_horizontal_texture_interpolation_via_stylizing_noise_earth_textures_new_test_resize512_aug_crop_128'; num_images = None
#interp_dir = '../naive_alpha_blending/output_horizontal_texture_interpolation_via_naive_alpha_blending_earth_textures_new_test_resize512_aug_crop_128'; num_images = None
train = EasyDict(func='util_scripts.interpolation_evaluation', src_dir=src_dir, interp_dir=interp_dir, num_images=num_images, minibatch_size=32*num_gpus, scale_h=1, scale_w=8)
'''





'''
env.CUDA_VISIBLE_DEVICES = '0,1,2,3'; num_gpus = 4
result_dir = 'results/swap_4_latentResEG_32_latentResDC_-1_latentChannels_128_genG_0.00_recG_1.00_pixel_200.00_perceptual_0.00_gram_0.0020_aCorr_0.00_classIm_0.00_classLatent_1.00_latent_0.00_KL_10.00_interpG_1.00_flickr_earth_textures_new_train_aug_crop_128_with_labels'
dataset.tfrecord_dir = 'flickr_earth_textures_new_test_aug_crop_128_with_labels'
train = EasyDict(func='util_scripts.generate_temporal_interpolation_video', run_id=0, snapshot=None, grid_size=[15,8], minibatch_size=64*num_gpus, duration_sec=10)
'''
#----------------------------------------------------------------------------
