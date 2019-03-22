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
    img_A = np.array(img_A)/127.5 - 1.
    img_B = load_image(depth_image_path)
    img_B = np.array(img_B)/127.5 - 1.
    img_AB = np.concatenate((img_A, img_B), axis=0)
    return img_AB


def load_image(image_path):
    input_img = imread(image_path)

    return input_img

# -----------------------------


def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)


def save_images(images, depth_images, size, image_path, loss_path):
    merged_images, losses = merge(inverse_transform(images), depth_images, size)
    f = open(loss_path, "w+")
    for loss in losses:
        f.write(str(loss))
        f.writelines("\n")
    f.close()
    return imsave(merged_images, image_path)


def imread(path, is_grayscale = True):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, depth_images, size):
    zipped = zip(images, depth_images)
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1] * 3, 3))
    losses = []
    for idx, image in enumerate(zipped):
        diff = image[1] - image[0]
        losses.append(np.sum(np.sum(diff)))
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image[0]
        img[j*h:j*h+h, i*w+w:i*w+w+w, :] = image[1]
        img[j*h:j*h+h, i*w+w+w:i*w+w+w+w, :] = diff
    return img, losses


def imsave(images, path):
    return scipy.misc.imsave(path, images)


def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.


def inverse_transform(images):
    return (images+1.)*127.5


def center_crop(x, ph, pw=None):
    if pw is None:
        pw = ph
    h, w = x.shape[:2]
    j = int(round((h - ph)/2.))
    i = int(round((w - pw)/2.))
    return x[j:j+ph, i:i+pw]
