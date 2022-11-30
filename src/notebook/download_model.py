#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

sys.path.append("../")


# In[ ]:


from base import SODModel, get_model
from utils import config_parser

config = config_parser()


# In[ ]:


robustbench = [
    "Wong2020Fast",
    "Engstrom2019Robustness",
    "Salman2020Do_R18",
    "Salman2020Do_R50",
    "Salman2020Do_50_2",
]
config.dataset = "imagenet"
config.device = "cpu"
for model in robustbench:
    get_model("robustbench", model, None, "../../storage/model/")


SODModel()
