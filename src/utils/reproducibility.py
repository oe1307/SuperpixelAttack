import os
import random

import numpy as np

from .logging import setup_logger

logger = setup_logger(__name__)


def reproducibility(seed=0, use_torch=True):
    """Set random seed for reproducibility."""

    logger.debug(f"[ REPRODUCIBILITY ] seed={seed}\n")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = f"{seed}"
    random.seed(seed)
    np.random.seed(seed)
    if use_torch:
        import torch

        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        logger.debug("torch seed is not set")
