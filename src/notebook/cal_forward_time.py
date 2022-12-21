#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import time

import robustbench
import torch
from robustbench.data import CustomImageFolder, get_preprocessing
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from torch import Tensor
from torch.utils.data import DataLoader

# In[ ]:


n_examples = 5000
batch_size = 1000
query = 100
thread = 5
device = 0
model_name = "Wong2020Fast"


# In[ ]:


def cw_loss(pred: Tensor, y: Tensor):
    pred_sorted, idx_sorted = pred.sort(dim=1, descending=True)
    class_pred = pred[torch.arange(pred.shape[0]), y]
    target_pred = torch.where(
        idx_sorted[:, 0] == y, pred_sorted[:, 1], pred_sorted[:, 0]
    )
    loss = target_pred - class_pred
    return loss


# In[ ]:


model = robustbench.load_model(model_name, "../../storage/model", "imagenet").to(device)
model.eval()
transform = get_preprocessing(
    BenchmarkDataset.imagenet, ThreatModel.Linf, model_name, None
)
dataset = CustomImageFolder("../../storage/data/imagenet", transform=transform)
dataloader = DataLoader(dataset, n_examples, shuffle=False, num_workers=thread)
print("Loading dataset...")
x_all, y_all = next(iter(dataloader))[:2]
x_all, y_all = x_all.to(device), y_all.to(device)
print("Loaded dataset")


# In[ ]:


cal_forward_time = 0
n_batch = math.ceil(n_examples / batch_size)
with torch.no_grad():
    for i in range(n_batch):
        for q in range(query):
            print(f"\r batch: {i}, forward: {q}     ", end="")
            start = i * batch_size
            end = min((i + 1) * batch_size, n_examples)
            x = x_all[start:end]
            y = y_all[start:end]
            timekeeper = time.time()
            pred = model(x).softmax(dim=1)
            loss = cw_loss(pred, y)
            cal_forward_time += time.time() - timekeeper

print(f"\ncalculate forward in {cal_forward_time:.2f} sec")
