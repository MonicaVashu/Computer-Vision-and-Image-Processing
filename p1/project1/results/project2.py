import cv2
import numpy as np
from scipy.spatial import distance

img1 = cv2.imread("D:/UB/summer/CVIP/p2/p1.jpg", 0)
img2 = cv2.imread("D:/UB/summer/CVIP/p2/p2.jpg", 0)
img3 = cv2.imread("D:/UB/summer/CVIP/p2/3.png")

sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(img1, None)
kp_2, desc_2 = sift.detectAndCompute(img2, None)
kp_3, desc_3 = sift.detectAndCompute(img3, None)

# Obtain key point pairs from two images
kp_final1 = []
kp_final2 = []
desc1 = []
desc2 = []
for i in range(desc_2.shape[0]):
    kp_new2 =[]
    min_d = float('Inf')
    for j in range(desc_1.shape[0]):
        dst = distance.hamming(desc_1[j], desc_2[i])
        if min_d > dst:
            min_d = dst
            kp_new2.append(kp_1[j])
            desc2.append(desc_2[i])
            desc1.append(desc_1[j])
    if min_d < 90:
        kp_final2.append(kp_2[i])
        kp_final1.append(kp_new2[-1])

cv2.drawKeypoints(img1, kp_final1, img1)
cv2.imwrite('sift_keypoints.jpg', img1)

cv2.drawKeypoints(img2, kp_final2, img2)
cv2.imwrite('sift_keypoints2.jpg', img2)

# Merge two images
col = img1.shape[1] + img2.shape[1]
row = img1.shape[0]
org_col = img1.shape[1]
cmp_row = img2.shape[1]
result = []
for i in range(row):
    row_elements = []
    for j in range(col):
        if j < org_col:
            row_elements.append(img1[i][j])
        else:
            row_elements.append(img2[i - row][j - org_col])
    result.append(row_elements)
result = np.array(result)
cv2.imwrite('mergedImage.jpg', np.array(result))
cv2.waitKey(0)

# Get co-ordinates of pair of key-points
x1 = np.zeros(len(kp_final2))
y1 = np.zeros(len(kp_final2))
x2 = np.zeros(len(kp_final2))
y2 = np.zeros(len(kp_final2))
I1 = []
I2 = []
for i in range(len(kp_final1)):
    I1.append(kp_final1[i].pt)
    x1[i], y1[i] = I1[i]
for i in range(len(kp_final2)):
    I2.append(kp_final2[i].pt)
    x2[i], y2[i] = I2[i]
x1 = x1.astype(int)
x2 = x2.astype(int)
y1 = y1.astype(int)
y2 = y2.astype(int)

# Thresholding to get imp key-point co-ordinates only
varx1 = []
vary1 = []
varx2 = []
vary2 = []
x = img1.shape[0]
img = cv2.imread("mergedImage.jpg")
for i in range(len(x1)):
    x_1 = x1[i]
    y_1 = y1[i]
    x_2 = x2[i]
    y_2 = y2[i]
    if (img1[x_1][y_1][1]) - (img2[x_2][y_2][1]) > 50:
        img = cv2.line(img, (x_1, y_1), (x+x_2, y_2), (255, 255, 255), 1)
        varx1.append(x_1)
        varx2.append(x_2)
        vary1.append(y_1)
        vary2.append(y_2)
cv2.imwrite("line.jpg", img)
cv2.waitKey(0)

# Leftmost point co-ordinates
I = 0
minimum = 2000
for i in range(len(vary1)):
    if vary1[i] < minimum:
        minimum = vary1[i]
        I = i
i = vary2[I]

diff = vary2[I] - vary1[I]

c = np.zeros(shape=(diff, img2.shape[1], 3))
print(c.shape)
print(img2[diff:, :].shape)
new = np.concatenate([img2[diff:, :], c], axis=0)
print(new.shape)

x = img1[:, :563]
y = new[:, 160:]
m = np.concatenate([x, y], axis=1)
cv2.imwrite("m.jpg", m)
cv2.waitKey(0)