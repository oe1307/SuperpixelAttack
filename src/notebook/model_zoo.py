#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from robustbench.model_zoo import models
from robustbench.utils import ThreatModel

# In[ ]:


list(models.cifar_10_models[ThreatModel.Linf].keys())


# In[ ]:


list(models.cifar_100_models[ThreatModel.Linf].keys())


# In[ ]:


list(models.imagenet_models[ThreatModel.Linf].keys())


# In[ ]:
