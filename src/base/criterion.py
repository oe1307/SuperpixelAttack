from typing import Callable

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from utils import config_parser

config = config_parser()


def get_criterion() -> Callable[[Tensor, Tensor], Tensor]:
    if config.criterion == "ce":
        return CrossEntropyLoss(reduction="none")
    elif config.criterion == "cw":
        return CWLoss()
    elif config.criterion == "dlr":
        return DLRLoss()
    raise NotImplementedError


class CWLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, y: Tensor) -> Tensor:
        r"""
        .. math::
            loss = max_{c \neq y}{f_c(x)} - f_y(x)
        """
        pred_sorted, idx_sorted = pred.sort(dim=1, descending=True)
        class_pred = pred[torch.arange(pred.shape[0]), y]
        target_pred = torch.where(
            idx_sorted[:, 0] == y, pred_sorted[:, 1], pred_sorted[:, 0]
        )
        loss = target_pred - class_pred
        return loss


class DLRLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, y: Tensor) -> Tensor:
        r"""
        .. math::
            loss = (max_{c \neq y}{f_c(x)} - f_y(x)) /
                        (max_{1st}{f_c(x)} - max_{3rd}{f_c(x)})
        """
        pred_sorted, idx_sorted = pred.sort(dim=1, descending=True)
        class_pred = pred[torch.arange(pred.shape[0]), y]
        target_pred = torch.where(
            idx_sorted[:, 0] == y, pred_sorted[:, 1], pred_sorted[:, 0]
        )
        loss = (target_pred - class_pred) / (pred_sorted[:, 0] - pred_sorted[:, 2])
        return loss
