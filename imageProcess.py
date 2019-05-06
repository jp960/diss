import cv2
import numpy as np
from matplotlib import pyplot as plt

org_img = cv2.imread('/home/janhavi/Documents/Final Year/DISS/data/images/basement_0001.png', 1)
seg_img = cv2.imread('/home/janhavi/Documents/Final Year/DISS/data/segmented/seg_basement_0001.png', 1)

# Normalise images
org_img_norm = cv2.normalize(org_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)

# Random Forests
structured_edges = np.zeros((480, 640, 1), dtype=np.float32)
detector = cv2.ximgproc.createStructuredEdgeDetection('model.yml')
detector.detectEdges(org_img_norm, structured_edges)

output_img_norm = cv2.normalize(structured_edges, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
output_img_norm_inverted = cv2.bitwise_not(output_img_norm)
new_seg = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)

# Sobel
line_sobel = cv2.Sobel(org_img, cv2.CV_8U, 1, 0, ksize=3)
output_sobel_norm = cv2.normalize(line_sobel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
line_sobel_inverted = cv2.bitwise_not(line_sobel)
overlap_sobel = cv2.addWeighted(seg_img, 0.3, line_sobel_inverted, 0.7, 0)

# Canny
edges = np.zeros((480, 640, 3), dtype=np.float32)
edges = cv2.Canny(org_img, 100, 200, edges)
edges_inverted = cv2.bitwise_not(edges)

overlap_canny = cv2.addWeighted(new_seg, 0.3, edges_inverted, 0.7, 0)
overlap_canny_add_output = cv2.addWeighted(edges_inverted, 0.3, output_img_norm_inverted, 0.7, 0)
overlap_canny_add_output_seg = cv2.addWeighted(new_seg, 0.2, overlap_canny_add_output, 0.8, 0)

# Plot original and final output
plt.subplot(1, 2, 1), plt.imshow(org_img, cmap = 'gray')
plt.title('', fontsize=25), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(overlap_canny_add_output, cmap = 'gray')
plt.title('', fontsize=25), plt.xticks([]), plt.yticks([])

# Plot all options

plt.subplot(2, 4, 1), plt.imshow(org_img, cmap = 'gray')  # Original
plt.title('A', fontsize=25), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 2), plt.imshow(seg_img, cmap = 'gray')  # Segmented image
plt.title('B', fontsize=25), plt.xticks([]), plt.yticks([])  # Sobel CV_8U inverted
plt.subplot(2, 4, 3), plt.imshow(line_sobel_inverted, cmap = 'gray')
plt.title('C', fontsize=25), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 4), plt.imshow(overlap_sobel, cmap = 'gray')  # Overlayed with Sobel
plt.title('D', fontsize=25), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 5), plt.imshow(edges_inverted, cmap = 'gray')  # Canny
plt.title('E', fontsize=25), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 6), plt.imshow(output_img_norm_inverted, cmap = 'gray')  # Random Forests Image
plt.title('F', fontsize=25), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 7), plt.imshow(overlap_canny_add_output, cmap = 'gray')  # Final Output
plt.title('G', fontsize=25), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 8), plt.imshow(overlap_canny_add_output_seg, cmap = 'gray')  # Overlayed with Random Forests + Canny
plt.title('H', fontsize=25), plt.xticks([]), plt.yticks([])

plt.show()
