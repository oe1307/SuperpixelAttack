from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from utils import config_parser

config = config_parser()


def get_criterion() -> Callable[[Tensor, Tensor], Tensor]:
    if config.criterion == "cw":
        return CWLoss()
    raise NotImplementedError


class CWLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit: Tensor, y: Tensor) -> Tensor:
        r"""
        .. math::
            loss = max_{c \neq y}{f_c(x)} - f_y(x)
        """
        pred = F.softmax(logit, dim=1)
        pred_sorted, idx_sorted = pred.sort(dim=1, descending=True)
        class_pred = pred[torch.arange(pred.shape[0]), y]
        target_pred = torch.where(
            idx_sorted[:, 0] == y, pred_sorted[:, 1], pred_sorted[:, 0]
        )
        loss = target_pred - class_pred
        return loss
