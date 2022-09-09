import os
import random

import numpy as np


def reproducibility(use_torch=True):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    np.random.seed(0)
    if use_torch:
        import torch

        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
