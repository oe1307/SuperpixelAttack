#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt

# In[ ]:


x_adv = np.load("/data1/issa/AdEx_BlackBox/result/square_imagenet_easy_5000.npy")
for i in range(10):
    plt.subplots(figsize=(6, 6))
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.imshow(x_adv[i].transpose(1, 2, 0))
    plt.show()
    plt.close()


# In[ ]:
