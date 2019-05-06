import os
import cv2
import numpy as np
from glob import glob


def get_filtered_image(img):
    img_norm = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)

    structured_edges = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
    detector = cv2.ximgproc.createStructuredEdgeDetection('model.yml')
    detector.detectEdges(img_norm, structured_edges)

    structured_edges_norm = cv2.normalize(structured_edges, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U)
    structured_edges_norm_inverted = cv2.bitwise_not(structured_edges_norm)

    edges = np.zeros(img.shape, dtype=np.uint8)
    edges = cv2.Canny(img, 100, 200, edges)
    edges_inverted = cv2.bitwise_not(edges)

    return cv2.addWeighted(edges_inverted, 0.3, structured_edges_norm_inverted, 0.7, 0)


if __name__ == '__main__':
    img_dir = "/home/janhavi/Documents/Final Year/DISS/data/SUNRGBD/kv2/kinect2data/"
    output_dir = "/home/janhavi/PycharmProjects/diss/SUNRGBD/preprocessed/"
    depths_dir = "/home/janhavi/PycharmProjects/diss/SUNRGBD/depths/"
    count = 0
    files = os.listdir(img_dir)
    files.sort()
    for folder in files:
        img_name = folder.split('_')[0]
        image_filepath = glob(os.path.join(img_dir+folder+"/image/*.jpg"))
        depth_filepath = glob(os.path.join(img_dir+folder+"/depth_bfx/*.png"))
        if len(image_filepath) > 0 and os.path.exists(image_filepath[0]):
            image = cv2.imread(image_filepath[0], 1)
            filtered_image = get_filtered_image(image)
            resized_image = cv2.resize(filtered_image, (256, 256))
            cv2.imwrite(output_dir+img_name+".png", resized_image)
        else:
            print(os.path.join(img_dir+folder+"/image/"))

        if len(depth_filepath) > 0 and os.path.exists(depth_filepath[0]):
            depth_image = cv2.imread(depth_filepath[0], 1)
            resized_depth_image = cv2.resize(depth_image, (256, 256))
            cv2.imwrite(depths_dir+img_name+".png", resized_depth_image)
        else:
            print(os.path.join(img_dir+folder+"/depth_bfx/"))

