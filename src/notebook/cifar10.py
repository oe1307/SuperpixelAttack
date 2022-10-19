#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

# In[4]:


cifar10 = json.load(open("../base/cifar10.json"))
for k in cifar10:
    print(k, len(cifar10[k]))


# In[ ]:
