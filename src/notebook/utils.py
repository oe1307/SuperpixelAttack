#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from robustbench.model_zoo import model_dicts
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from utils import config_parser

config = config_parser()


def check_model(model_name, data_dir):
    models = model_dicts[BenchmarkDataset(config.dataset)]
    models = models[ThreatModel(config.norm)].keys()
    assert (
        model_name in models
    ), f"{model_name} not in robustbench[{config.dataset}][{config.norm}]"
