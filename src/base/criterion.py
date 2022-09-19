from typing import Callable

import torch
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
    elif config.criterion == "fitness":
        return fitness
    else:
        raise NotImplementedError


def cw_loss(logits: Tensor, y: Tensor) -> Tensor:
    r"""
    .. math::
        loss = max_{i \neq y}z_i - z_y
    """
    logits_sorted, idx_sorted = logits.sort(dim=1, descending=True)
    class_logits = logits[torch.arange(logits.shape[0]), y]
    target_logits = torch.where(
        idx_sorted[:, 0] == y, logits_sorted[:, 1], logits_sorted[:, 0]
    )
    loss = target_logits - class_logits
    return loss


def dlr_loss(logits: Tensor, y: Tensor) -> Tensor:
    raise NotImplementedError("dlr_loss is not implemented yet.")


def fitness(logits: Tensor, y: Tensor) -> Tensor:
    """fitness function for GenAttack"""
    logits_sorted, idx_sorted = logits.sort(dim=1, descending=True)
    target_logits = torch.where(
        idx_sorted[:, 0] == y, logits_sorted[:, 1], logits_sorted[:, 0]
    )
    other_logits = torch.log(torch.exp(logits).sum(dim=1) - torch.exp(target_logits))
    return target_logits - other_logits
