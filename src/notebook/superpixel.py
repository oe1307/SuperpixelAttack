#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

sys.path.append("..")


# In[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

from base._dataset import load_imagenet
from utils import config_parser

# In[ ]:


config = config_parser()
config.norm = "Linf"
data, label = load_imagenet("Salman2020Do_R18", "../../storage/data")


# In[ ]:


# 元データ
x = data.numpy()[1]
print(x.shape)
x = (x.transpose(1, 2, 0) * 255).astype(np.uint8)
plt.subplots(figsize=(6, 6))
plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
plt.imshow(x)
plt.show()
plt.close()


# In[ ]:


# パラメータ
region_size = 20
ruler = 30
min_element_size = 20
num_iterations = 4

converted = cv2.cvtColor(x, cv2.COLOR_RGB2HSV_FULL)
slic = cv2.ximgproc.createSuperpixelSLIC(
    converted, cv2.ximgproc.MSLIC, region_size, float(ruler)
)
slic.iterate(num_iterations)
slic.enforceLabelConnectivity(min_element_size)
result = x.copy()
contour_mask = slic.getLabelContourMask(False)
result[0 < contour_mask] = (0, 255, 255)
plt.subplots(figsize=(6, 6))
plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
plt.imshow(result)
plt.show()
plt.close()


# In[ ]:


x = cv2.imread("/data1/issa/AdEx_BlackBox/result/tmp.jpg")
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255
np.save("/data1/issa/AdEx_BlackBox/result/x_2.npy", x)
x.shape


# In[ ]:


# 元データ
x = np.load("/data1/issa/AdEx_BlackBox/result/x_2.npy")
print(x.shape)
x = (x.transpose(1, 2, 0) * 255).astype(np.uint8)
plt.subplots(figsize=(6, 6))
plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
plt.imshow(x)
plt.show()
plt.close()


# In[ ]:


# パラメータ
region_size = 20
ruler = 30
min_element_size = 20
num_iterations = 4

converted = cv2.cvtColor(x, cv2.COLOR_RGB2HSV_FULL)
slic = cv2.ximgproc.createSuperpixelSLIC(
    converted, cv2.ximgproc.MSLIC, region_size, float(ruler)
)
slic.iterate(num_iterations)
slic.enforceLabelConnectivity(min_element_size)
result = x.copy()
contour_mask = slic.getLabelContourMask(False)
result[0 < contour_mask] = (0, 255, 255)
plt.subplots(figsize=(6, 6))
plt.imshow(result)
plt.show()
plt.close()


# In[ ]:


# パラメータ
region_size = 20
ruler = 30
min_element_size = 10
num_iterations = 4

# 分割数
stac = []
for i in range(100):
    x = data.numpy()[i]
    x = (x.transpose(1, 2, 0) * 255).astype(np.uint8)
    converted = cv2.cvtColor(x, cv2.COLOR_RGB2HSV_FULL)
    slic = cv2.ximgproc.createSuperpixelSLIC(
        converted, cv2.ximgproc.MSLIC, region_size, float(ruler)
    )
    slic.iterate(num_iterations)
    slic.enforceLabelConnectivity(min_element_size)
    nseg = slic.getNumberOfSuperpixels()
    stac.append(nseg)
print(stac, "\n")
print(np.average(stac))
print(np.max(stac), np.min(stac))
print(np.where(stac == np.min(stac)))


# In[ ]:


# 画像のサイズ
print(data[0].shape)
3 * 224 * 224


# In[ ]:
