import random
from math import ceil

import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter, map_coordinates, rotate

quantiles = np.sort(np.random.normal(0, 1, 64 * 64))


def quantile_normalize(img, distribution=quantiles):
    order = np.argsort(img.flatten())
    location = np.argsort(order)
    return np.take_along_axis(distribution, location, 0).reshape((64, 64))


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    m = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, m, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


class YeastDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, X, Y, batch_size=32, augment=False):
        self.X = X
        self.Y = Y
        self.augment = augment
        self.batch_size = batch_size

    def __len__(self):
        return ceil(len(self.X) / self.batch_size)

    def __getitem__(self, item):
        batch_x = self.X.take(range(item * self.batch_size, (item + 1) * self.batch_size), mode="wrap", axis=0)
        batch_labels = self.Y.take(range(item * self.batch_size, (item + 1) * self.batch_size), mode="wrap", axis=0)
        batch_y = np.zeros((self.batch_size, 3))
        for i in range(self.batch_size):
            label = batch_labels[i]  # get label
            batch_y[i, label] = 1  # set corresponding label to 1 (one-hot encoded)
            # for chan in range(2):
            #   batch_x[i, :, :, chan] = quantile_normalize(batch_x[i, :, :, chan].reshape((32, 32)))

            if self.augment:
                # 90 degree rotations
                rotation = random.choice([0, 90, 180, 270])  # rotation amount
                batch_x[i] = rotate(batch_x[i], rotation, axes=(1, 0))

                # elastic deformations. Params selected by visual inspection
                rand_int = random.randint(1, 142857)
                for chan in range(2):
                    batch_x[i, :, :, chan] = elastic_transform(batch_x[i, :, :, chan].reshape((64, 64)), 60, 15, 1,
                                                               random_state=np.random.RandomState(rand_int))

        return batch_x, batch_y


class YeastDataGeneratorBinary(tf.keras.utils.Sequence):

    def __init__(self, X, Y, batch_size=32, augment=False):
        self.X = X
        self.Y = Y
        self.augment = augment
        self.batch_size = batch_size

    def __len__(self):
        return ceil(len(self.X) / self.batch_size)

    def __getitem__(self, item):
        batch_x = self.X.take(range(item * self.batch_size, (item + 1) * self.batch_size), mode="wrap", axis=0)
        batch_y = self.Y.take(range(item * self.batch_size, (item + 1) * self.batch_size), mode="wrap", axis=0)
        for i in range(self.batch_size):

            #for chan in range(2):
                #batch_x[i, :, :, chan] = quantile_normalize(batch_x[i, :, :, chan].reshape((64, 64)))

            if self.augment:
                # 90 degree rotations
                rotation = random.choice([0, 90, 180, 270])  # rotation amount
                batch_x[i] = rotate(batch_x[i], rotation, axes=(1, 0))

                # elastic deformations. Params selected by visual inspection

                rand_int = random.randint(1, 142857)
                for chan in range(2):
                    batch_x[i, :, :, chan] = elastic_transform(batch_x[i, :, :, chan].reshape((64, 64)), 60, 15, 1,
                                                               random_state=np.random.RandomState(rand_int))

        return batch_x, batch_y
