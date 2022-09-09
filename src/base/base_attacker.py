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
        self.best_loss = torch.zeros((self.config.n_examples, self.config.iteration))
        self.current_loss = torch.zeros((self.config.n_examples, self.config.iteration))
        self.num_forward = 0
        self.num_backward = 0

    @torch.inference_mode()
    def attack(self, model, data, label, criterion):
        self.savedir = self.config.savedir + f"{model.name}"
        os.makedirs(self.savedir)

        num_batch = math.ceil(data.shape[0] / model.batch_size)
        for i in range(num_batch):
            start = i * model.batch_size
            end = min((i + 1) * model.batch_size, self.config.n_examples)
            x = data[start:end].to(self.config.device)
            y = label[start:end].to(self.config.device)
            breakpoint()
            self._attack(model, x, y, criterion)

        self.record()

    def record(self):
        self.total_time = time.time() - self.total_time
        self.clean_acc /= self.config.n_examples
        self.robust_acc /= self.config.n_examples
        self.ASR = 1 - self.robust_acc

        np.savetxt(f"{self.savedir}/best_loss.csv", self.best_loss)
        np.savetxt(f"{self.savedir}/current_loss.csv", self.current_loss)

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
