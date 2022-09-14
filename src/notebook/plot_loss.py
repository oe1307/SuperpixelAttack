#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt


# In[ ]:


dirname = "../../result/APGD/Carmon2019Unlabeled/2022-09-14_20:14/"


# In[ ]:


best_loss = np.load(dirname + "best_loss.npy")
average_best_loss = np.average(best_loss, axis=0)
plt.subplots(figsize=(8, 8))
plt.plot(average_best_loss, label="best loss")
plt.legend()
plt.show()
plt.close()


# In[ ]:


current_loss = np.load(dirname + "current_loss.npy")
plt.subplots(figsize=(8, 8))
for idx in range(3):
    plt.plot(current_loss[idx])
plt.show()
plt.close()


# In[ ]:


step_size = np.load(dirname + "step_size.npy")
plt.subplots(figsize=(8, 8))
for idx in range(step_size.shape[0]):
    plt.plot(step_size[idx])
plt.show()
plt.close()


# In[ ]:


best_loss = np.load(dirname + "best_loss.npy")
average_best_loss = np.average(best_loss, axis=0)
plt.subplots(figsize=(8, 8))
plt.plot(average_best_loss, label="best loss")
plt.legend()
plt.show()
plt.close()
