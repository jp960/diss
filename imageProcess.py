import cv2
import numpy as np
from matplotlib import pyplot as plt

org_img = cv2.imread('images/basement_0001.png', 1)
seg_img = cv2.imread('segmented/seg_basement_0001.png', 1)

org_img_norm = cv2.normalize(org_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)

structured_edges = np.zeros((480, 640, 1), dtype=np.float32)
detector = cv2.ximgproc.createStructuredEdgeDetection('model.yml')
detector.detectEdges(org_img_norm, structured_edges)

output_img_norm = cv2.normalize(structured_edges, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
output_img_norm_inverted = cv2.bitwise_not(output_img_norm)
new_seg = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)

# # Output dtype = cv2.CV_8U
line_sobel = cv2.Sobel(org_img, cv2.CV_8U, 1, 0, ksize=3)
output_sobel_norm = cv2.normalize(line_sobel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
line_sobel_inverted = cv2.bitwise_not(line_sobel)
overlap_sobel = cv2.addWeighted(seg_img, 0.3, line_sobel_inverted, 0.7, 0)

edges = np.zeros((480, 640, 3), dtype=np.float32)
edges = cv2.Canny(org_img, 100, 200, edges)
edges_inverted = cv2.bitwise_not(edges)
overlap_canny = cv2.addWeighted(new_seg, 0.3, edges_inverted, 0.7, 0)
overlap_canny_add_output = cv2.addWeighted(edges_inverted, 0.3, output_img_norm_inverted, 0.7, 0)
overlap_canny_add_output_seg = cv2.addWeighted(new_seg, 0.2, overlap_canny_add_output, 0.8, 0)

plt.subplot(2, 4, 1), plt.imshow(org_img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 2), plt.imshow(seg_img, cmap = 'gray')
plt.title('Segmented image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 3), plt.imshow(line_sobel_inverted, cmap = 'gray')
plt.title('Sobel CV_8U inverted'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 4), plt.imshow(overlap_sobel, cmap = 'gray')
plt.title('Overlayed with Sobel'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 5), plt.imshow(edges_inverted, cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 6), plt.imshow(output_img_norm_inverted, cmap = 'gray')
plt.title('Random Forests Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 7), plt.imshow(overlap_canny_add_output, cmap = 'gray')
plt.title('Random Forests + Canny'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 8), plt.imshow(overlap_canny_add_output_seg, cmap = 'gray')
plt.title('Overlayed with Random Forests + Canny'), plt.xticks([]), plt.yticks([])

plt.show()