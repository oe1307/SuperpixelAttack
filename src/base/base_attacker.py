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
        self.savedir = config.savedir + f"{model.name}"
        os.makedirs(self.savedir)

        num_batch = math.ceil(data.shape[0] / model.batch_size)
        for i in range(num_batch):
            logger.info(f"Batch {i + 1}/{num_batch}")
            start = i * model.batch_size
            end = min((i + 1) * model.batch_size, config.n_examples)
            x = data[start:end].to(config.device)
            y = label[start:end].to(config.device)
            self._clean_acc(model, x, y, criterion, start, end)
            self._attack(model, x, y, criterion, start, end)

        self.record()

    @torch.inference_mode()
    def _clean_acc(self, model, x: Tensor, y: Tensor, criterion, start: int, end: int):
        logits = model(x).clone().detach()
        self.clean_acc += torch.sum(logits.argmax(dim=1) == y).item()
        loss = criterion(logits, y).clone().detach()
        self.best_loss[start:end, 0] = loss
        self.current_loss[start:end, 0] = loss
        self.num_forward += x.shape[0]

    @torch.no_grad()
    def _robust_acc(
        self, logits: Tensor, y: Tensor, loss: Tensor, start: int, end: int, iter: int
    ) -> Tensor:
        loss = loss.clone().detach()
        self.robust_acc += torch.sum(logits.argmax(dim=1) == y).item()
        self.current_loss[start:end, iter + 1] = loss
        self.best_loss[start:end, iter + 1] = torch.max(
            loss, self.best_loss[start:end, iter]
        )
        self.num_forward += logits.shape[0]
