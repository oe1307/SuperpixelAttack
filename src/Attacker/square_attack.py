import math
import time

import torch
import torchattacks
from torch import Tensor

from base import Attacker
from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class SquareAttack(Attacker):
    def __init__(self):
        config.n_forward = config.restart * config.step

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        attacker = TorchAttackSquare(
            self.model,
            config.norm,
            config.epsilon,
            config.step,
            config.restart,
            config.p_init,
            seed=config.seed,
        )
        x_adv_all = []
        n_images = x_all.shape[0]
        n_batch = math.ceil(n_images / config.batch_size)
        for i in range(n_batch):
            pbar.debug(i + 1, n_batch, "batch")
            start = i * config.batch_size
            end = min((i + 1) * config.batch_size, n_images)
            x = x_all[start:end]
            y = y_all[start:end]
            x_adv = attacker.perturb(x, y)
            x_adv_all.append(x_adv)
        x_adv_all = torch.cat(x_adv_all, dim=0)
        return x_adv_all


class TorchAttackSquare(torchattacks.Square):
    def init_hyperparam(self, x):
        assert self.norm in ["Linf", "L2"]
        assert self.eps is not None
        assert self.loss in ["ce", "margin", "cw", "dlr"]

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def margin_and_loss(self, x, y):
        if config.criterion == "ce":
            return super().margin_and_loss(x, y)
        elif config.criterion == "cw":
            return self.cw_loss(x, y)
        elif config.criterion == "dlr":
            return self.dlr_loss(x, y)

    def cw_loss(self, x, y):
        logits = self.get_logits(x)
        logits_sorted, idx_sorted = logits.sort(dim=1, descending=True)
        class_pred = logits[torch.arange(logits.shape[0]), y]
        target_pred = torch.where(
            idx_sorted[:, 0] == y, logits_sorted[:, 1], logits_sorted[:, 0]
        )
        loss = class_pred - target_pred
        return loss, loss

    def dlr_loss(self, x, y):
        logits = self.get_logits(x)
        logits_sorted, idx_sorted = logits.sort(dim=1, descending=True)
        class_pred = logits[torch.arange(logits.shape[0]), y]
        target_pred = torch.where(
            idx_sorted[:, 0] == y, logits_sorted[:, 1], logits_sorted[:, 0]
        )
        loss = (class_pred - target_pred) / (logits_sorted[:, 0] - logits_sorted[:, 2])
        return loss, loss
