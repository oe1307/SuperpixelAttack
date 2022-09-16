#!/usr/bin/env python
# coding: utf-8

# In[25]:


import sys

sys.path.append("../")


# In[45]:


import robustbench
import torch

from attacker import get_attacker
from base import get_criterion
from utils import config_parser, setup_logger

# In[46]:


logger = setup_logger.setLevel(10)


# In[47]:


config = config_parser()

config_parser.config.device = 0
config_parser.config.criterion = "cw"
config_parser.config.attacker = "APGD"
config_parser.config.n_examples = 10
config_parser.config.iteration = 100
config_parser.config.epsilon = 8 / 255
config_parser.config.step_size = 4 / 255
config_parser.config.alpha = 0.75
config_parser.config.rho = 0.75
# config_parser.config.dataset = "cifar10"
# config_parser.config.norm = "Linf"


# In[48]:


data, label = robustbench.load_cifar10(10, "../../storage/data/")
data, label = data.to(config.device), label.to(config.device)
model = robustbench.load_model("Carmon2019Unlabeled", "../../storage/model/").to(
    config.device
)
model.eval()
model.batch_size = 100
criterion = get_criterion()

attacker = get_attacker()
attacker.record = lambda: None
attacker.attack(model, data, label, criterion)


# In[ ]:


# In[ ]:
