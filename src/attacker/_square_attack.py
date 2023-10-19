import math
import time

import torch
import torchattacks
from torch import Tensor

from base import Attacker
from utils import ProgressBar, config_parser

config = config_parser()


class SquareAttack(Attacker):
    def _attack(self, data: Tensor, label: Tensor) -> Tensor:
        attacker = TorchAttackSquare(
            self.model,
            config.norm,
            config.epsilon,
            config.iter,
            config.restart,
            config.p_init,
            config.criterion,
            seed=config.seed,
        )
        adv_data = []
        n_batch = math.ceil(len(data) / config.batch_size)
        pbar = ProgressBar(n_batch, "batch", color="cyan")
        for b in range(n_batch):
            start = b * config.batch_size
            end = min((b + 1) * config.batch_size, len(data))
            x, y = data[start:end], label[start:end]
            upper = (x + config.epsilon).clamp(0, 1)
            lower = (x - config.epsilon).clamp(0, 1)
            x_adv = attacker.perturb(x.to(config.device), y).cpu()
            x_adv = x_adv.clamp(lower, upper)
            adv_data.append(x_adv)
            pbar.step()
        adv_data = torch.cat(adv_data, dim=0)
        pbar.end()
        return adv_data


class TorchAttackSquare(torchattacks.Square):
    def init_hyperparam(self, x):
        assert self.norm in ("Linf", "L2")
        assert self.eps is not None
        assert self.loss in ("ce", "margin", "cw", "dlr")

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def margin_and_loss(self, x, y):
        if config.criterion in ("ce", "margin"):
            return super().margin_and_loss(x, y)
        elif config.criterion == "cw":
            self.loss = "margin"
            return super().margin_and_loss(x, y)
        elif config.criterion == "dlr":
            return self.dlr_loss(x, y)
        else:
            raise ValueError(f"Unknown criterion: {config.criterion}")

    def dlr_loss(self, x, y):
        logits = self.get_logits(x)
        logits_sorted, idx_sorted = logits.sort(dim=1, descending=True)
        class_prediction = logits[torch.arange(logits.shape[0]), y]
        target_prediction = torch.where(
            idx_sorted[:, 0] == y, logits_sorted[:, 1], logits_sorted[:, 0]
        )
        loss = (class_prediction - target_prediction) / (
            logits_sorted[:, 0] - logits_sorted[:, 2]
        )
        return loss, loss
