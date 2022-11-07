import math
import os
import time

import torch
from torch import Tensor

from Utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class Attacker:
    def __init__(self):
        self.timekeeper = time.time()
        self.robust_acc = 0

    @torch.no_grad()
    def attack(self, model, data: Tensor, label: Tensor):
        assert not model.training
        self.model = model

        n_batch = math.ceil(data.shape[0] / model.batch_size)
        for i in range(n_batch):
            pbar(i + 1, n_batch)
            self.start = i * model.batch_size
            self.end = min((i + 1) * model.batch_size, config.n_examples)
            x = data[self.start : self.end].to(config.device)
            y = label[self.start : self.end].to(config.device)
            x_adv = self._attack(x, y)

            upper = (x + config.epsilon).clamp(0, 1).clone()
            lower = (x - config.epsilon).clamp(0, 1).clone()
            assert torch.all(x_adv <= upper + 1e-10)
            assert torch.all(x_adv >= lower - 1e-10)
            logits = self.model(x_adv).clone()
            self.robust_acc += (logits.argmax(dim=1) == y).sum().item()
            logger.info(f"Robust accuracy : {self.robust_acc} / {self.end}")
            torch.cuda.empty_cache()

        total_n_forward = data.shape[0] * self.n_forward
        total_time = time.time() - self.timekeeper
        robust_acc = self.robust_acc / data.shape[0] * 100
        ASR = 100 - robust_acc

        os.makedirs(f"../result/{config.attacker}", exist_ok=True)
        msg = ""
        for k, v in config.items():
            msg += f"{k} = {v}\n"
        print(msg, file=open(f"../result/{config.attacker}/{config.datetime}.txt", "w"))

        msg = (
            +"\n"
            + f"num_img = {self.end}\n"
            + f"total time (sec) = {total_time:.2f}s\n"
            + f"robust acc (%) = {robust_acc:.2f}\n"
            + f"ASR (%) = {ASR:.2f}\n"
            + f"n_forward = {self.n_forward}\n"
            + f"total n_forward = {total_n_forward}"
        )
        print(msg, file=open(f"../result/{config.attacker}/{config.datetime}.txt", "a"))
        logger.info(msg)
