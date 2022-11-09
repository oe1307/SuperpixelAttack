#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from matplotlib import pyplot as plt

# In[14]:


pgd_success_iter = np.load("/data1/issa/AdEx_BlackBox/result/PGD_success_iter.npy")
print(pgd_success_iter.min())
print(pgd_success_iter.max())
acc = []
for i in range(101):
    acc.append((pgd_success_iter <= i).sum())


# In[33]:


plt.subplots(figsize=(8, 8))
plt.plot(acc)
plt.scatter(0, acc[0], label="0 iteration")
plt.text(0 + 2, acc[0], f"{acc[0]}")
plt.scatter(1, acc[1], label="1 iteration")
plt.text(1 + 2, acc[1], f"{acc[1]}")
plt.scatter(2, acc[2], label="2 iterations")
plt.text(2 + 2, acc[2], f"{acc[2]}")
plt.scatter(3, acc[3], label="3 iterations")
plt.text(3 + 2, acc[3], f"{acc[3]}")
plt.scatter(100, acc[100], label="100 iterations")
plt.text(100 - 5, acc[100] - 100, f"{acc[100]}")
plt.xlabel("Iteration")
plt.ylabel("Number of successful attacks")
plt.legend()
plt.show()
plt.close()


# In[ ]:
