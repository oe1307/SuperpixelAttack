import math
import time

import torch
from torch import Tensor

from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class Attacker:
    def __init__(self):
        self.timekeeper = time.time()
        self.robust_acc = 0

    def attack(self, model, data: Tensor, label: Tensor):
        self.model = model

        num_batch = math.ceil(data.shape[0] / model.batch_size)
        for i in range(num_batch):
            self.start = i * model.batch_size
            self.end = min((i + 1) * model.batch_size, config.n_examples)
            x = data[self.start : self.end].to(config.device)
            y = label[self.start : self.end].to(config.device)
            self._attack(x, y)
            torch.cuda.empty_cache()

        total_num_forward = data.shape[0] * self.num_forward
        total_time = time.time() - self.timekeeper
        robust_acc = self.robust_acc / data.shape[0] * 100
        ASR = 100 - robust_acc

        msg = (
            "\n"
            + f"num_img = {self.end}\n"
            + f"total time (sec) = {total_time:.2f}s\n"
            + f"robust acc (%) = {robust_acc:.2f}\n"
            + f"ASR (%) = {ASR:.2f}\n"
            + f"num_forward = {self.num_forward}\n"
            + f"total num_backward = {total_num_forward}"
        )

        print(msg, file=open(config.savedir + "/summary.txt", "w"))
        logger.info(msg)
