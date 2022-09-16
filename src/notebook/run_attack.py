#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

sys.path.append("../")


# In[ ]:


from datetime import datetime

import torch
from torchvision import transforms as T

from attacker import get_attacker
from base import get_criterion, get_model, load_dataset, load_imagenet
from utils import config_parser, setup_logger

# In[ ]:


config = config_parser()
logger = setup_logger.setLevel(10)

model_name = "Wong2020Fast"
model_container = "robustbench"
idx = 0

config_parser.config.attacker = "HALS"
config_parser.config.dataset = "imagenet"
config_parser.config.norm = "Linf"
config_parser.config.device = 1
config_parser.config.criterion = "cw"
config_parser.config.n_examples = 10
config_parser.config.epsilon = 8 / 255
config_parser.config.batch_size = 100

config_parser.config.iteration = 10
config_parser.config.max_iter = 1
config_parser.config.step_size = 16 / 255
config_parser.config.alpha = 0.75
config_parser.config.rho = 0.75
config_parser.config.initial_split = 32


# In[ ]:


data, label = load_dataset(model_name, "../../storage/data/")
data = data[idx].unsqueeze(0).to(config.device)
label = label[idx].unsqueeze(0).to(config.device)
model = get_model(
    model_container, model_name, config.batch_size, model_dir="../../storage/model"
)
criterion = get_criterion()


# In[ ]:


attacker = get_attacker()
attacker.iter, attacker.start, attacker.end = 0, 0, 1
attacker.model = model
attacker.criterion = criterion


# In[ ]:


upper = (data + config.epsilon).clamp(0, 1)
lower = (data - config.epsilon).clamp(0, 1)
attacker.clean_acc(data, label)
x_adv = attacker._attack(data, label).detach().clone()


# In[ ]:


to_pillow = T.ToPILImage()
to_pillow(x_adv[0])


# In[ ]:


to_pillow(data[0])


# In[ ]:


to_pillow(x_adv[0] - data[0])


# In[ ]:


is_upper = (x_adv[0] >= (upper[0] - 0.0001)).sum().item()
is_lower = (x_adv[0] <= (lower[0] + 0.0001)).sum().item()
print(f"num bound = {is_upper + is_lower} / {x_adv.numel()}")


# In[ ]:
