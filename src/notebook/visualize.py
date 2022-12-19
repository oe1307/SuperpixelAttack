#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

sys.path.append("..")


# In[ ]:


import random

from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries, slic

from Attacker.UpdateArea import Superpixel
from base import get_model, load_imagenet
from utils import config_parser

# In[ ]:


config = config_parser()
config.norm = "Linf"
config.device = "cpu"
config.n_examples = 1
config.thread = 10
model, transform = get_model("Salman2020Do_R18", None)
data, label = load_imagenet(transform, "../../storage/data/")


# In[ ]:


config.segments = [4, 16, 64]
superpixel_manager = Superpixel()
superpixel = superpixel_manager._cal_superpixel(data[0], 1, 1)


# In[ ]:


plt.subplots(figsize=(8, 8))
plt.axis("off")
plt.imshow(mark_boundaries(data[0].numpy().transpose(1, 2, 0), superpixel[0]))
plt.show()


# In[ ]:


plt.subplots(figsize=(8, 8))
plt.axis("off")
plt.imshow(superpixel[2])
plt.show()


# In[ ]:


plt.subplots(figsize=(8, 8))
plt.axis("off")
plt.imshow(data[0].numpy().transpose(1, 2, 0))
plt.show()


# In[ ]:


epsilon = 0.1
x_adv = data[0].numpy().copy()
for i in range(1, superpixel[0].max() + 1):
    x_adv[random.randint(0, 2), superpixel[0] == i] += (
        2 * random.randint(0, 1) - 1
    ) * epsilon
x_adv = x_adv.clip(0, 1)

plt.subplots(figsize=(8, 8))
plt.axis("off")
plt.imshow(mark_boundaries(x_adv.transpose(1, 2, 0), superpixel[0]))
plt.show()

for i in range(1, superpixel[1].max() + 1):
    x_adv[random.randint(0, 2), superpixel[1] == i] += (
        2 * random.randint(0, 1) - 1
    ) * epsilon
x_adv = x_adv.clip(0, 1)

plt.subplots(figsize=(8, 8))
plt.axis("off")
plt.imshow(mark_boundaries(x_adv.transpose(1, 2, 0), superpixel[1]))
plt.show()


# In[ ]:
