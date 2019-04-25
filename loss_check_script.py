import math
import numpy as np
import scipy.misc
from utils import *
from glob import glob

if __name__ == '__main__':
    preprocessed_data = np.random.choice(glob('/home/janhavi/PycharmProjects/diss/SUNRGBD/preprocessed/*.png'), 5)
    depth_data = [path.replace('preprocessed', 'depths') for path in preprocessed_data]
    depth_images_raw = [load_image(path) for path in depth_data]
    data = list(zip(preprocessed_data, depth_data))
    sample = [load_data(sample_file[0], sample_file[1]) for sample_file in data]

    depth_images = np.array(depth_images_raw).astype(np.float32)[:, :, :, None]
    sample_images = np.array(sample).astype(np.float32)[:, :, :, None]

    samples = np.zeros(depth_images.shape)
    for i in range(5):
        samples[i, :, :, :] = sample_images[i, :256, :, :]
    #     plt.subplot(1, 5, i+1), plt.imshow(samples[i, :, :, 0], cmap='gray')
    # plt.show()

    save_images(samples, depth_images, [5, 1], 1, 1)

