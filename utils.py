"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import tensorflow as tf
import numpy as np
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix


def load_data(image_path, depth_image_path, flip=True, is_test=False):
    img_A = load_image(image_path)
    print(img_A.shape)
    img_B = load_image(depth_image_path)
    img_AB = np.concatenate((img_A, img_B), axis=0)
    # img_AB.reshape((512, 256, 1))
    return img_AB


def load_image(image_path):
    input_img = imread(image_path)

    return input_img

# -----------------------------


def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, is_grayscale = True):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.


def center_crop(x, ph, pw=None):
    if pw is None:
        pw = ph
    h, w = x.shape[:2]
    j = int(round((h - ph)/2.))
    i = int(round((w - pw)/2.))
    return x[j:j+ph, i:i+pw]
