#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from robustbench.model_zoo import models
from robustbench.utils import ThreatModel

# In[ ]:


print("\n" + "-" * 10 + " cifar10 " + "-" * 10)
for model in models.cifar_10_models[ThreatModel.Linf].keys():
    print(model)


# In[ ]:


print("\n" + "-" * 10 + " cifar100 " + "-" * 10)
for model in models.cifar_100_models[ThreatModel.Linf].keys():
    print(model)


# In[ ]:


print("\n" + "-" * 10 + " imagenet " + "-" * 10)
for model in models.imagenet_models[ThreatModel.Linf].keys():
    print(model)


# In[ ]:
