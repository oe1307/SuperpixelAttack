import math
from datetime import datetime as dt

import torch
from torch import Tensor

from utils import config_parser, setup_logger

from .recorder import Recorder

logger = setup_logger(__name__)
config = config_parser()


class Attacker(Recorder):
    def __init__(self):
        super().__init__()

    def attack(self, model, data, label, criterion):
        self.model = model
        self.criterion = criterion

        num_batch = math.ceil(data.shape[0] / model.batch_size)
        for i in range(num_batch):
            self.start = i * model.batch_size
            self.end = min((i + 1) * model.batch_size, config.n_examples)
            x = data[self.start : self.end].to(config.device)
            y = label[self.start : self.end].to(config.device)
            self.iter = 0
            self.clean_acc(x, y)
            self._attack(x, y)
            logger.warning(f"Robust accuracy : {self._robust_acc.sum()} / {self.end}")
            torch.cuda.empty_cache()
            self.record()
        logger.warning(f"Attack end at {dt.now().strftime('%Y/%m/%d %H:%M:%S')}")

    @torch.inference_mode()
    def clean_acc(self, x: Tensor, y: Tensor):
        logits = self.model(x).detach().clone()
        acc = logits.argmax(dim=1) == y
        self._clean_acc += acc.sum().item()
        self._robust_acc[self.start : self.end] = acc
        self.success_iter[self.start : self.end] = acc
        loss = self.criterion(logits, y).detach().clone()
        self.best_loss[self.start : self.end, 0] = loss
        self.current_loss[self.start : self.end, 0] = loss
        logger.warning(f"Clean accuracy : {self._clean_acc} / {self.end}")

    def robust_acc(self, x_adv: Tensor, y: Tensor) -> Tensor:
        self.iter += 1
        logits = self.model(x_adv).clone()
        self.num_forward += x_adv.shape[0]
        self._robust_acc[self.start : self.end] = torch.logical_and(
            self._robust_acc[self.start : self.end], logits.argmax(dim=1) == y
        )
        self.success_iter[self.start : self.end] += self._robust_acc[
            self.start : self.end
        ]
        loss = self.criterion(logits, y).clone()
        self.current_loss[self.start : self.end, self.iter] = loss.detach().clone()
        self.best_loss[self.start : self.end, self.iter] = torch.max(
            loss.detach().clone(), self.best_loss[self.start : self.end, self.iter - 1]
        )
        logger.debug(
            f"Robust accuracy ( iter={self.iter} ) :"
            + f" {self._robust_acc.sum()} / {self.end}"
        )
        return loss
