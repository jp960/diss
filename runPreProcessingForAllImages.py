import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from shutil import copy2
from glob import glob


def get_filtered_image(img):
    img_norm = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)

    structured_edges = np.zeros((427, 561, 1), dtype=np.float32)
    detector = cv2.ximgproc.createStructuredEdgeDetection('model.yml')
    detector.detectEdges(img_norm, structured_edges)

    structured_edges_norm = cv2.normalize(structured_edges, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U)
    structured_edges_norm_inverted = cv2.bitwise_not(structured_edges_norm)

    edges = np.zeros((427, 561, 3), dtype=np.float32)
    edges = cv2.Canny(img, 100, 200, edges)
    edges_inverted = cv2.bitwise_not(edges)
    return cv2.addWeighted(edges_inverted, 0.3, structured_edges_norm_inverted, 0.7, 0)


if __name__ == '__main__':
    # img_dir = "/home/janhavi/Documents/Final Year/DISS/data/SUNRGBD/kv1/NYUdata/"
    # output_dir = "/home/janhavi/PycharmProjects/diss/NYU/preprocessed/"
    # depths_dir = "/home/janhavi/PycharmProjects/diss/NYU/depths/"
    # count = 0
    # for folder in os.listdir(img_dir):
    #     image_filepath = os.path.join(img_dir+folder+"/image/"+folder+".jpg")
    #     depth_filepath = os.path.join(img_dir+folder+"/depth/"+folder+".png")
    #     if os.path.exists(image_filepath):
    #         image = cv2.imread(image_filepath, 1)
    #         filtered_image = get_filtered_image(image)
    #         resized_image = cv2.resize(filtered_image, (256, 256))
    #         cv2.imwrite(output_dir+folder+".png", resized_image)
    #
    #     if os.path.exists(depth_filepath):
    #         depth_image = cv2.imread(depth_filepath, 1)
    #         resized_depth_image = cv2.resize(depth_image, (256, 256))
    #         cv2.imwrite(depths_dir+folder+".png", resized_depth_image)

    data_pre = sorted(glob('/home/janhavi/PycharmProjects/diss/NYU/preprocessed/*.png'))
    data_ny = sorted(glob('/home/janhavi/PycharmProjects/diss/NYU/depths/*.png'))
    data = list(zip(data_pre, data_ny))
    print(len(data_ny))
    print(len(data))
