import os
import cv2
from matplotlib import pyplot as plt


if __name__ == '__main__':
    img_dir = "/home/janhavi/Documents/Final Year/DISS/data/images/"
    seg_dir = "/home/janhavi/Documents/Final Year/DISS/data/segmented/"
    output_dir = "/home/janhavi/Documents/Final Year/DISS/data/preprocessed/"
    count = 0
    for filename in os.listdir(img_dir):
        print('seg_' + filename)

    test = cv2.imread(img_dir + 'basement_0001.png', 0)
    plt.subplot(1, 1, 1), plt.imshow(test, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.show()

    cv2.imwrite(output_dir + 'basement_0001.png', test)
