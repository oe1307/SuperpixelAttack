#!/usr/bin/env python
# coding: utf-8

# In[16]:


import matplotlib.pyplot as plt
import numpy as np

# In[17]:


upper = np.load("/data1/issa/AdEx_BlackBox/result/cifar10_upper.npy")
lower = np.load("/data1/issa/AdEx_BlackBox/result/cifar10_lower.npy")
x_adv_10 = np.load("/data1/issa/AdEx_BlackBox/result/square_adv_10.npy")
x_adv_100 = np.load("/data1/issa/AdEx_BlackBox/result/square_adv_100.npy")
x_adv_1000 = np.load("/data1/issa/AdEx_BlackBox/result/square_adv_1000.npy")
x_adv_5000 = np.load("/data1/issa/AdEx_BlackBox/result/square_cifar10_5000.npy")


# In[18]:


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


# In[19]:


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


# In[20]:


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


# In[21]:


print(x_adv_5000.shape)
print(upper.shape)
print(lower.shape)
num_bound = []
for x_adv, idx_upper, idx_lower in zip(x_adv_5000, upper, lower):
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


# In[31]:


import bisect
import math


def _get_percentage_of_elements(i_iter: int) -> float:
    i_p = i_iter / 10000
    intervals = [0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    p_ratio = [0.5**i for i in range(len(intervals) + 1)]
    i_ratio = bisect.bisect_left(intervals, i_p)
    return 0.8 * p_ratio[i_ratio]


# In[40]:


height, width = 32, 32
n_elements = []
for i_iter in range(10000):
    percentage_of_elements = _get_percentage_of_elements(i_iter)
    height_tile = max(int(round(math.sqrt(percentage_of_elements * height * width))), 1)
    print(height_tile)
    n_elements.append(height_tile * height_tile * 3)


plt.subplots(figsize=(8, 8))
plt.plot(n_elements)
plt.xlabel("Number of iterations")
plt.ylabel("Number of pixels")
plt.title("Square attack on CIFAR-10")
plt.show()


# In[37]:


n_elements[11]


# In[ ]:
