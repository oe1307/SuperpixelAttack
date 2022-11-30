#!/usr/bin/env python
# coding: utf-8

# In[5]:


from glob import glob

# In[27]:


for result in glob("/data1/issa/AdEx_BlackBox/result/imagenet/*/summary.txt"):
    summary = open(result).readlines()
    print(
        summary[2].split(" ")[-1].strip(),
        summary[3].split(" ")[-1].strip(),
        summary[4].split(" ")[-1].strip(),
        summary[-2].split(" ")[-1].strip() + "%",
    )


# In[25]:


print(summary)


# In[ ]:
