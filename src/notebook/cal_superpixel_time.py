#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from robustbench.data import CustomImageFolder
from skimage.segmentation import slic
from torch.utils.data import DataLoader
from torchvision import transforms as T

# In[ ]:


n_examples = 5000
thread = 10
segments = [4, 16, 64, 256]


# In[ ]:


def cal_superpixel(x, idx):
    print(f"\r {idx + 1} / {n_examples}", end="")
    superpixel_storage = []
    for n_segments in segments:
        img = (x.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        superpixel = slic(img, n_segments=n_segments)
        superpixel_storage.append(superpixel)
    return superpixel_storage


# In[ ]:


transform = T.Compose([T.Resize(232), T.CenterCrop(224), T.ToTensor()])
dataset = CustomImageFolder("../../storage/data/imagenet", transform=transform)
print("Loading dataset...")
dataloader = DataLoader(dataset, n_examples, shuffle=False, num_workers=thread)
print("Loaded dataset")
img = next(iter(dataloader))[0]


# In[ ]:


timekeeper = time.time()

if thread == 1:
    superpixel = np.array([cal_superpixel(img[idx], idx) for idx in range(n_examples)])
else:
    with ThreadPoolExecutor(thread) as executor:
        futures = [
            executor.submit(cal_superpixel, img[idx], idx) for idx in range(n_examples)
        ]
    superpixel = np.array([future.result() for future in futures])

print(f"calculate superpixel in {time.time() - timekeeper:.2f} sec")
