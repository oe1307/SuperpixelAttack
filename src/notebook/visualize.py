#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

sys.path.append("..")


# In[ ]:


from matplotlib import pyplot as plt

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
plt.imshow(superpixel[1])
plt.show()


# In[ ]:


plt.subplots(figsize=(8, 8))
plt.axis("off")
plt.imshow(data[0].numpy().transpose(1, 2, 0))
plt.show()


# In[ ]:


data[0].shape


# In[ ]:
