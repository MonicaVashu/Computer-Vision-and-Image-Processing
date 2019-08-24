import cv2
import numpy as np

original = cv2.imread("D:/UB/summer/CVIP/p1/project1/m1.jpg", 0)
image_to_compare = cv2.imread("D:/UB/summer/CVIP/p1/project1/m2.jpg", 0)

sift = cv2.xfeatures2d.SURF_create(7000)
kp_1, desc_1 = sift.detectAndCompute(original, None)
kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

original_withKP = cv2.drawKeypoints(original, kp_1, outImage=np.array([]))
cv2.imwrite("D:/UB/summer/CVIP/p2/original_withKP.jpg", original_withKP)

image_to_compare_withKP = cv2.drawKeypoints(image_to_compare, kp_1, outImage=np.array([]))
cv2.imwrite("D:/UB/summer/CVIP/p2/image_to_compare_withKP.jpg", image_to_compare_withKP)

bf = cv2.BFMatcher()
matches = bf.knnMatch(desc_1, desc_2, k=1)

draw_params = dict(matchColor=(0,255,0), singlePointColor=None, flags=2)
img3 = cv2.drawMatchesKnn(original, kp_1, image_to_compare, kp_2, matches, None, **draw_params)
cv2.imwrite("original_image_drawMatches.jpg", img3)

stitcher = cv2.createStitcher(False)
# foo = cv2.imread("D:/foo.png")
# bar = cv2.imread("D:/bar.png")
result = stitcher.stitch((original,image_to_compare))

cv2.imwrite("stitched.jpg", result)