import math
import os
import time

import numpy as np
import torch

from utils import config_parser, setup_logger

logger = setup_logger(__name__)


class Attacker:
    def __init__(self):
        self.config = config_parser.config
        self.recorder()

    def recorder(self):
        self.total_time = time.time()
        self.clean_acc = 0
        self.robust_acc = 0
        self.best_loss = torch.zeros(
            (self.config.n_examples, self.config.iteration + 1)
        )
        self.current_loss = torch.zeros(
            (self.config.n_examples, self.config.iteration + 1)
        )
        self.num_forward = 0
        self.num_backward = 0
        self._recorder()

    def _recorder(self):
        pass

    def attack(self, model, data, label, criterion):
        self.savedir = self.config.savedir + f"{model.name}"
        os.makedirs(self.savedir)

        num_batch = math.ceil(data.shape[0] / model.batch_size)
        for i in range(num_batch):
            logger.info(f"batch {i + 1}/{num_batch}")
            self.start = i * model.batch_size
            self.end = min((i + 1) * model.batch_size, self.config.n_examples)
            x = data[self.start : self.end].to(self.config.device)
            y = label[self.start : self.end].to(self.config.device)
            self._clean_acc(model, x, y, criterion)
            self._attack(model, x, y, criterion)

        self.record()

    @torch.inference_mode()
    def _clean_acc(self, model, x, y, criterion):
        logits = model(x).clone().detach()
        self.clean_acc += torch.sum(logits.argmax(dim=1) == y).item()
        loss = criterion(logits, y).detach().cpu()
        self.best_loss[self.start : self.end, 0] = loss
        self.current_loss[self.start : self.end, 0] = loss
        self.num_forward += x.shape[0]

    def record(self):
        self._record()

        self.total_time = time.time() - self.total_time
        self.clean_acc = self.clean_acc / self.config.n_examples * 100
        self.robust_acc = self.clean_acc / self.config.n_examples * 100
        self.ASR = 100 - self.robust_acc

        np.save(f"{self.savedir}/best_loss.npy", self.best_loss.cpu().numpy())
        np.save(f"{self.savedir}/current_loss.npy", self.best_loss.cpu().numpy())

        print(
            "\n"
            + f"total time (sec) = {self.total_time:.2f}s\n"
            + f"clean acc (%) = {self.clean_acc:.2f}\n"
            + f"robust acc (%) = {self.robust_acc:.2f}\n"
            + f"ASR (%) = {self.ASR:.2f}\n"
            + f"num_forward = {self.num_forward}\n"
            + f"num_backward = {self.num_backward}\n",
            file=open(self.savedir + "/summary.txt", "w"),
        )

    def _record(self):
        pass
