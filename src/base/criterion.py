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
