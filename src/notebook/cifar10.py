#!/usr/bin/env python
# coding: utf-8

# In[43]:


import json

import numpy as np

# In[44]:


success_iter = np.load(
    "/data1/issa/AdEx_BlackBox/result/FGSM/cifar10/Linf/Carmon2019Unlabeled/2022-10-14_15:19:04/success_iter.npy"
)


# In[45]:


index0 = np.where(success_iter == 0)  # 元々誤分類
index1 = np.where(success_iter == 1)  # 1回目で成功
index2 = np.where(np.logical_and(1 < success_iter, success_iter < 10))  # 2回目以降で成功
index3 = np.where(success_iter == 11)  # 失敗


# In[46]:


print(index0[0].shape)
print(index1[0].shape)
print(index2[0].shape)
print(index3[0].shape)


# In[47]:


index = dict()
index["clean"] = index0[0].tolist()
index["easy"] = index1[0].tolist()
index["hard"] = index2[0].tolist()
index["fail"] = index3[0].tolist()


# In[48]:


json.dump(index, open("../../storage/data/cifar10.json", "w"))


# In[ ]:
