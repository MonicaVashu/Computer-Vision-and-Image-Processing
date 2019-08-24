#!/usr/bin/env python
# coding: utf-8

# In[491]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import math
import pandas as pd


# In[492]:


original = cv2.imread("p1.jpg")
image_to_compare = cv2.imread("p2.jpg")


# In[493]:


# 2) Check for similarities between the 2 images
sift = cv2.xfeatures2d.SIFT_create(20)
kp_1, desc_1 = sift.detectAndCompute(original, None)
kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)


# In[494]:


from scipy.spatial import distance
min_kp = {}
kp_final1 =[]
kp_final2 =[]
desc1=[]
desc2=[]

for i in range(desc_2.shape[0]):
#     kp_new1.append(kp_1[i])
    kp_new2 =[]
    min_d = float('Inf')
    for j in range(desc_1.shape[0]):
        dst = distance.hamming(desc_1[j], desc_2[i])
        if min_d > dst:
            min_d = dst
            kp_new2.append(kp_1[j])
            desc2.append(desc_2[i])
            desc1.append(desc_1[j])
#             min_kp[kp_1[i]] = kp_2[j]
    if min_d < 90:
        kp_final2.append(kp_2[i])
        kp_final1.append(kp_new2[-1])


# In[495]:


cv2.drawKeypoints(original,kp_final1,original)
cv2.imwrite('sift_keypoints.jpg',original)


# In[496]:


cv2.drawKeypoints(image_to_compare,kp_final2,image_to_compare)
cv2.imwrite('sift_keypoints2.jpg',image_to_compare)


# In[497]:


col = original.shape[1]+image_to_compare.shape[1]
row = original.shape[0]

# print(row)
# print(col)
org_col=original.shape[1]
cmp_row=image_to_compare.shape[1]
result=[]

for i in range(row):
    row_elements = []
    for j in range(col):
        if j < org_col:
            row_elements.append(original[i][j])
        else:
            row_elements.append(image_to_compare[i-row][j-org_col])
    result.append(row_elements)
#print(result)


# In[498]:


result=np.array(result)
cv2.imwrite('mergedImage.jpg',np.array(result))
print(result.shape)


# In[499]:


x1=np.zeros(len(kp_final2))
y1=np.zeros(len(kp_final2))
x2=np.zeros(len(kp_final2))
y2=np.zeros(len(kp_final2))
I1=[]
I2=[]
for i in range(len(kp_final1)):
    I1.append(kp_final1[i].pt)
    x1[i],y1[i]=I1[i]
for i in range(len(kp_final2)):
    I2.append(kp_final2[i].pt)
    x2[i],y2[i]=I2[i]

x1=x1.astype(int)
x2=x2.astype(int)
y1=y1.astype(int)
y2=y2.astype(int)
x2


# In[500]:


img=cv2.imread("C:/Users/malin/mergedImage.jpg")
print(img.shape)
print(original.shape)
print(image_to_compare.shape)


# In[501]:


varx1=[]
vary1=[]
varx2=[]
vary2=[]
x=original.shape[0]
img=cv2.imread("C:/Users/malin/mergedImage.jpg")
for i in range(len(x1)):
    x_1=x1[i]
    y_1=y1[i]
    x_2=x2[i]
    y_2=y2[i]
    #for k in range(3):
    if(original[x_1][y_1][1]-image_to_compare[x_2][y_2][1]>50):
        img = cv2.line(img,(x_1,y_1),(x+x_2,y_2),(255,255,255),1)
        varx1.append(x_1)
        varx2.append(x_2)
        vary1.append(y_1)
        vary2.append(y_2)
cv2.imwrite("line.jpg",img)
cv2.waitKey(0)


# In[503]:


I=0
minimum=2000
for i in range(len(vary1)):
     if(vary1[i]<minimum):
            minimum=vary1[i]
            I=i
print(minimum)
print(I)
i=vary2[I]
#print(vary2[I])


# In[520]:


diff = vary2[I] - vary1[I]
print(diff)


# In[522]:


# image_to_compare = image_to_compare[diff:,:]
c = np.zeros(shape=(diff,image_to_compare.shape[1],3))
print(c.shape)
print(image_to_compare[diff:,:].shape)
new = np.concatenate([image_to_compare[diff:,:],c], axis = 0)
print(new.shape)


# In[524]:


x = original[:, :563]
y = new[:, 160:]
# x=cv2.imread("C:/Users/malin/x.jpg")
# y=cv2.imread("C:/Users/malin/y.jpg")

m = np.concatenate([x,y], axis = 1)
cv2.imwrite("m.jpg",m)
cv2.waitKey(0)


# In[ ]:


# r=original.shape[0]
# c=original.shape[1]+image_to_compare.shape[1]

# for i in range(r):
#     for j in range(c):
#         row_elements = []
#         if(i<c):
#             row_elements.append(original[i][j]) 


# In[339]:


# x=original.shape[0]
# img=cv2.imread("C:/Users/malin/mergedImage.jpg")
# for i in range(len(x1)):
#      img = cv2.line(img,(x1[i],y1[i]),(x+x2[i],y2[i]),(255,255,255),1)
# cv2.imwrite("line.jpg",img)
# cv2.waitKey(0)


# In[336]:


#img[1][1][1]


# In[ ]:


# #x=original.shape[0]
# #img=cv2.imread("C:/Users/malin/mergedImage.jpg")
# #for i in range(len(x1)):
# #    img = cv2.line(img,(x1[i],y1[i]),(x+x2[i],y2[i]),(255,255,255),1)
# cv2.imwrite("line.jpg",img)
# cv2.waitKey(0)

