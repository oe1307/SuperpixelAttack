from typing import Callable

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from utils import config_parser

config = config_parser()


def get_criterion() -> Callable[[Tensor, Tensor], Tensor]:
    if config.criterion == "cw":
        return CWLoss()
    elif config.criterion == "ce":
        return CrossEntropyLoss(reduction="none")
    elif config.criterion == "dlr":
        return DLRLoss()
    elif config.criterion == "fitness":
        return Fitness()
    else:
        raise NotImplementedError


class CWLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
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


class DLRLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError("DLRLoss is not implemented yet.")


class Fitness(Module):
    """fitness function for GenAttack"""

    def __init__(self):
        super().__init__()

    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        idx_sorted = logits.sort(dim=1, descending=True)[1]
        target = torch.where(idx_sorted[:, 0] == y, idx_sorted[:, 1], idx_sorted[:, 0])
        first = logits[range(logits.shape[0]), target]
        second = torch.log(torch.exp(logits).sum(1) - first)
        return first - second
