import math
import os

import torch
from torch import Tensor

from utils import config_parser, setup_logger

from .recorder import Recorder

logger = setup_logger(__name__)


class Attacker(Recorder):
    def __init__(self):
        super().__init__()

    def attack(self, model, data, label, criterion):
        config = config_parser.config
        self.model = model
        self.criterion = criterion
        self.savedir = config.savedir + f"{model.name}"
        os.makedirs(self.savedir)

        num_batch = math.ceil(data.shape[0] / model.batch_size)
        for i in range(num_batch):
            self.start = i * model.batch_size
            self.end = min((i + 1) * model.batch_size, config.n_examples)
            x = data[self.start : self.end].to(config.device)
            y = label[self.start : self.end].to(config.device)
            self.iter = 0
            self.clean_acc(x, y)
            self._attack(x, y)
        self.record()

    @torch.inference_mode()
    def clean_acc(self, x: Tensor, y: Tensor):
        logits = self.model(x).detach().clone()
        self._robust_acc[self.start : self.end] = logits.argmax(dim=1) == y
        self._clean_acc += self._robust_acc.sum().item()
        loss = self.criterion(logits, y).detach().clone()
        self.best_loss[self.start : self.end, 0] = loss
        self.current_loss[self.start : self.end, 0] = loss
        self.num_forward += x.shape[0]
        logger.debug(f"Clean accuracy: {self._clean_acc} / {self.end}")

    def robust_acc(self, x_adv: Tensor, y: Tensor) -> Tensor:
        self.iter += 1
        logits = self.model(x_adv).clone()
        self._robust_acc[self.start : self.end] = torch.logical_and(
            self._robust_acc[self.start : self.end], logits.argmax(dim=1) == y
        )
        loss = self.criterion(logits, y).clone()
        self.current_loss[self.start : self.end, self.iter] = loss.detach().clone()
        self.best_loss[self.start : self.end, self.iter] = torch.max(
            loss.detach().clone(), self.best_loss[self.start : self.end, self.iter - 1]
        )
        self.num_forward += self.end - self.start
        logger.debug(
            f"Robust accuracy ( iter={self.iter} ):"
            + f" {self._robust_acc.sum()} / {self.end}"
        )
        return loss
