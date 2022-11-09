#!/usr/bin/env python
# coding: utf-8

# In[32]:


import matplotlib.pyplot as plt
import numpy as np

# In[33]:


upper = np.load("/data1/issa/AdEx_BlackBox/result/cifar10_upper.npy")
lower = np.load("/data1/issa/AdEx_BlackBox/result/cifar10_lower.npy")
x_adv_10 = np.load("/data1/issa/AdEx_BlackBox/result/square_adv_10.npy")
x_adv_100 = np.load("/data1/issa/AdEx_BlackBox/result/square_adv_100.npy")
x_adv_1000 = np.load("/data1/issa/AdEx_BlackBox/result/square_adv_1000.npy")


# In[34]:


print(x_adv_10.shape)
print(upper.shape)
print(lower.shape)
num_bound = []
for x_adv, idx_upper, idx_lower in zip(x_adv_10, upper, lower):
    num_bound.append((x_adv == idx_lower).sum() + (x_adv == idx_upper).sum())
    # num_bound.append((np.logical_or(x_adv == idx_lower, x_adv == idx_upper)).sum())
num_bound = np.array(num_bound) / 3072 * 100
len_bar = []
for i in range(0, 110, 10):
    # print(i, i + 10)
    len_bar.append((np.logical_and(i <= num_bound, num_bound < i + 10)).sum())
plt.subplots(figsize=(8, 8))
plt.bar(np.arange(5, 115, 10), len_bar, width=10)
plt.xlabel(
    "Percentage of pixels in the adversarial image that are bounded by the bounds"
)
plt.ylabel("Number of adversarial images")
plt.title("Square attack on CIFAR-10")
plt.xticks(np.arange(0, 110, 10))
plt.show()
plt.close()


# In[35]:


print(x_adv_100.shape)
print(upper.shape)
print(lower.shape)
num_bound = []
for x_adv, idx_upper, idx_lower in zip(x_adv_100, upper, lower):
    num_bound.append((x_adv == idx_lower).sum() + (x_adv == idx_upper).sum())
    # num_bound.append((np.logical_or(x_adv == idx_lower, x_adv == idx_upper)).sum())
num_bound = np.array(num_bound) / 3072 * 100
len_bar = []
for i in range(0, 110, 10):
    # print(i, i + 10)
    len_bar.append((np.logical_and(i <= num_bound, num_bound < i + 10)).sum())
plt.subplots(figsize=(8, 8))
plt.bar(np.arange(5, 115, 10), len_bar, width=10)
plt.xlabel(
    "Percentage of pixels in the adversarial image that are bounded by the bounds"
)
plt.ylabel("Number of adversarial images")
plt.title("Square attack on CIFAR-10")
plt.xticks(np.arange(0, 110, 10))
plt.show()
plt.close()


# In[36]:


print(x_adv_1000.shape)
print(upper.shape)
print(lower.shape)
num_bound = []
for x_adv, idx_upper, idx_lower in zip(x_adv_1000, upper, lower):
    num_bound.append((x_adv == idx_lower).sum() + (x_adv == idx_upper).sum())
    # num_bound.append((np.logical_or(x_adv == idx_lower, x_adv == idx_upper)).sum())
num_bound = np.array(num_bound) / 3072 * 100
len_bar = []
for i in range(0, 110, 10):
    # print(i, i + 10)
    len_bar.append((np.logical_and(i <= num_bound, num_bound < i + 10)).sum())
plt.subplots(figsize=(8, 8))
plt.bar(np.arange(5, 115, 10), len_bar, width=10)
plt.xlabel(
    "Percentage of pixels in the adversarial image that are bounded by the bounds"
)
plt.ylabel("Number of adversarial images")
plt.title("Square attack on CIFAR-10")
plt.xticks(np.arange(0, 110, 10))
plt.show()
plt.close()


# In[ ]:
