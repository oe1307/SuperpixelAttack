#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from glob import glob

from matplotlib import pyplot as plt

# Single

# In[ ]:


file = "../"


# In[ ]:


plt.subplots(figsize=(8, 8))


# Multi

# In[ ]:


files = glob(file + "/*.npy")


# In[ ]:


plt.subplots(figsize=(8, 8))


# In[ ]:
