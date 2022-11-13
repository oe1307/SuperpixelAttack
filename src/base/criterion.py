from typing import Callable

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from utils import config_parser

config = config_parser()


def get_criterion() -> Callable[[Tensor, Tensor], Tensor]:
    if config.criterion == "ce":
        return CrossEntropyLoss(reduction="none")
    else:
        raise NotImplementedError
