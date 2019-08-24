#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


Input = cv2.imread('C:/Users/malin/OneDrive/Desktop/CVIP/project1/project1/data/t2.jpg')
c = cv2.imread('C:/Users/malin/OneDrive/Desktop/CVIP/project1/project1/data/c.jpg')
match = 0


# In[3]:


for i in range(1,Input.shape[0]-10):
    for j in range(1,Input.shape[1]-10):
        for k in range(c.shape[0]):
            for m in range(c.shape[1]):
                if (c[k][m].all() == Input[i-1+k][j-1+m].all()):
                    match = match + 1
    


# In[4]:


print("c_row",c.shape[0])
print("c_col",c.shape[1])
print("img_row",Input.shape[0])
print("img_col",Input.shape[1])


# In[5]:


points = []
c=np.array(c)
Input=np.array(Input)
for i in range(1,Input.shape[0]-10):
    for j in range(1,Input.shape[1]-10):
        for k in range(c.shape[0]):
            for m in range(c.shape[1]):
                if (c[k][m] == Input[i-1+k][j-1+m]):
                    #match = match + 1
                    points.append((j,i))
print(points)
    


# In[ ]:


points


# In[ ]:


img = np.array(Input)
for pt in (points):
    cv2.rectangle(img, pt, (pt[0] + c.shape[0], pt[1] + c.shape[1]), (0, 255, 255),2)

cv2.imwrite("C:/Users/malin/OneDrive/Desktop/CVIP/project1/project1/results/x.jpg",img)

