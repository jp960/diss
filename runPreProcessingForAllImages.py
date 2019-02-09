import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_filtered_image(img):
    img_norm = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)

    structured_edges = np.zeros((480, 640, 1), dtype=np.float32)
    detector = cv2.ximgproc.createStructuredEdgeDetection('model.yml')
    detector.detectEdges(img_norm, structured_edges)

    structured_edges_norm = cv2.normalize(structured_edges, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U)
    structured_edges_norm_inverted = cv2.bitwise_not(structured_edges_norm)

    edges = np.zeros((480, 640, 3), dtype=np.float32)
    edges = cv2.Canny(img, 100, 200, edges)
    edges_inverted = cv2.bitwise_not(edges)
    return cv2.addWeighted(edges_inverted, 0.3, structured_edges_norm_inverted, 0.7, 0)


if __name__ == '__main__':
    img_dir = "/home/janhavi/PycharmProjects/diss/images/"
    seg_dir = "/home/janhavi/PycharmProjects/diss/segmented/"
    output_dir = "/home/janhavi/PycharmProjects/diss/preprocessed/"
    count = 0
    for filename in os.listdir(img_dir):
        image = cv2.imread(img_dir+filename, 1)
        filtered_image = get_filtered_image(image)
        resized_image = cv2.resize(filtered_image, (256, 256))
        cv2.imwrite(output_dir+filename, resized_image)
        count += 1
        print("done: " + filename + str(count))
