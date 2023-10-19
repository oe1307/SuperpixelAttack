import os
import random
import traceback
from functools import wraps

from .color_print import ProgressBar, printc
from .config_parser import config_parser

config = config_parser()

__all__ = [
    "printc",
    "ProgressBar",
    "config_parser",
]


def fix_seed(seed=0, use_numpy=True, use_torch=True):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if use_numpy:
        import numpy as np

        np.random.seed(seed)
    if use_torch:
        import torch

        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)  # raise error if non-deterministic


def save_error():
    def _save_error(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as error:
                print(
                    traceback.format_exc(),
                    file=open(f"{config.savedir}/error.txt", "a+"),
                )
                raise error

        return wrapper

    return _save_error
