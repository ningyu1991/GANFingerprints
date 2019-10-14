import numpy as np
from PIL import Image
import chainer
import random
import scipy.misc


class LSUNBedroom200kDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, root, size=128, resize_method='bilinear', augmentation=False, crop_ratio=1.0):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.size = size
        self.resize_method = resize_method
        self.augmentation = augmentation
        self.crop_ratio = crop_ratio

    def __len__(self):
        return len(self.base)

    def transform(self, image):
        image = image / 128. - 1.
        image += np.random.uniform(size=image.shape, low=0., high=1. / 128)
        return image

    def get_example(self, i):
        image, label = self.base[i]
        image = self.transform(image)
        return image, label


if __name__ == "__main__":
    import glob, os, sys

    root_path = sys.argv[1]

    count = 0
    n_image_list = []
    filenames = glob.glob(root_path + '/*.png')
    for filename in filenames:
        filename = filename.split('/')
        n_image_list.append([filename[-1], 0])
        count += 1
        if count % 10000 == 0:
            print(count)
    print("Num of examples:{}".format(count))
    n_image_list = np.array(n_image_list, np.str)
    np.savetxt('lsun_bedroom_200k_png_image_list.txt', n_image_list, fmt="%s")