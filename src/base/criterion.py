from typing import Callable

import numpy as np
from torch import Tensor
from torch.nn import CrossEntropyLoss

from utils import config_parser

config = config_parser()


def get_criterion() -> Callable[[Tensor, Tensor], Tensor]:
    if config.criterion == "cw":
        return cw_loss
    elif config.criterion == "ce":
        return CrossEntropyLoss(reduction="none")
    elif config.criterion == "dlr":
        return dlr_loss
    else:
        raise NotImplementedError


def cw_loss(logits: Tensor, y: Tensor) -> Tensor:
    r"""
    .. math::
        loss = max_{i \neq y}z_i - z_y
    """
    logits_sorted, idx_sorted = logits.sort(dim=1)
    acc = idx_sorted[:, -1] == y
    z_y = logits[np.arange(logits.shape[0]), y]
    max_zi = logits_sorted[:, -2] * acc + logits_sorted[:, -1] * ~acc
    loss = max_zi - z_y
    return loss


def dlr_loss(logits: Tensor, y: Tensor) -> Tensor:
    raise NotImplementedError("dlr_loss is not implemented yet.")
