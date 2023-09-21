import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss as CEloss
from torch.nn import Module

from utils import config_parser

config = config_parser()


def get_criterion() -> Module:
    if config.criterion not in criterions:
        raise NotImplementedError(config.criterion)
    return criterions[config.criterion]


class CWloss(Module):
    def forward(self, prediction: Tensor, y: Tensor) -> Tensor:
        """
        loss = max_{c != y}{f_c(x)} - f_y(x)
        """
        prediction_sorted, idx_sorted = prediction.sort(dim=1, descending=True)
        class_prediction = prediction[torch.arange(prediction.shape[0]), y]
        target_prediction = torch.where(
            idx_sorted[:, 0] == y, prediction_sorted[:, 1], prediction_sorted[:, 0]
        )
        loss = target_prediction - class_prediction
        return loss


class DLRloss(Module):
    def forward(self, prediction: Tensor, y: Tensor) -> Tensor:
        """
        loss = (max_{c != y}{f_c(x)} - f_y(x)) /
                    (max_{1st}{f_c(x)} - max_{3rd}{f_c(x)})
        """
        prediction_sorted, idx_sorted = prediction.sort(dim=1, descending=True)
        class_prediction = prediction[torch.arange(prediction.shape[0]), y]
        target_prediction = torch.where(
            idx_sorted[:, 0] == y, prediction_sorted[:, 1], prediction_sorted[:, 0]
        )
        loss = (target_prediction - class_prediction) / (
            prediction_sorted[:, 0] - prediction_sorted[:, 2]
        )
        return loss


criterions = {
    "ce": CEloss(reduction="none"),
    "cw": CWloss(),
    "dlr": DLRloss(),
}
