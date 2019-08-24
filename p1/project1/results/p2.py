import cv2
import numpy as np

original = cv2.imread("D:/UB/summer/CVIP/p1/project1/m1.jpg", 0)
image_to_compare = cv2.imread("D:/UB/summer/CVIP/p1/project1/m2.jpg", 0)

sift = cv2.xfeatures2d.SIFT_create(7000)
kp_1, desc_1 = sift.detectAndCompute(original, None)
kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

# from scipy.spatial import distance
# desc_1_neighbours = []
# for i in range(len(desc_1)):
#     temp = []
#     for j in desc_2:
#         temp.append(distance.euclidean(desc_1[i], j))
#     desc_1_neighbours.append(min(temp))
# print(desc_1_neighbours)


original_withKP = cv2.drawKeypoints(original, kp_1, outImage=np.array([]))
cv2.imwrite("D:/UB/summer/CVIP/p2/original_withKP.jpg", original_withKP)

image_to_compare_withKP = cv2.drawKeypoints(image_to_compare, kp_1, outImage=np.array([]))
cv2.imwrite("D:/UB/summer/CVIP/p2/image_to_compare_withKP.jpg", image_to_compare_withKP)

bf = cv2.BFMatcher()
matches = bf.knnMatch(desc_1, desc_2, k=1)

# good = []
# for m in matches:
#     if m[0].distance < 0.5*m[1].distance:
#         good.append(m)
# matches = np.asarray(good)

# if len(matches[:,0]) >= 4:
#     src = np.float32([ kp_1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
#     dst = np.float32([ kp_2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
#     H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

# import matplotlib.pyplot as plt
# dst = cv2.warpPerspective(image_to_compare,H,(original.shape[1] + image_to_compare.shape[1], original.shape[0]))
# plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
# plt.show()
# plt.figure()
# dst[0:original.shape[0], 0:original.shape[1]] = original
# cv2.imwrite('output.jpg', dst)
# plt.imshow(dst)
# plt.show()

draw_params = dict(matchColor=(0,255,0), singlePointColor=None, flags=2)
img3 = cv2.drawMatchesKnn(original, kp_1, image_to_compare, kp_2, matches, None, **draw_params)
cv2.imwrite("original_image_drawMatches.jpg", img3)