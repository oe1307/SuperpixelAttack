import os
import time

import torch
from torch import Tensor

from utils import config_parser, rename_dir, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class Attacker:
    @torch.no_grad()
    def attack(self, model, data: Tensor, label: Tensor):
        assert not model.training
        self.model = model
        self.robust_acc = torch.zeros(data.shape[0], dtype=torch.bool)
        config.savedir = rename_dir(f"../result/{config.dataset}/{config.attacker}")
        os.makedirs(config.savedir, exist_ok=True)
        config.save()

        self._attack(data, label)

        torch.cuda.empty_cache()
        total_time = time.time() - self.timekeeper
        robust_acc = self.robust_acc.sum() / data.shape[0] * 100
        ASR = 100 - robust_acc

        print(
            "\n"
            + f"n_img = {self.end}\n"
            + f"total time (sec) = {total_time:.2f}s\n"
            + f"ASR (%) = {ASR:.2f}\n",
            file=open(f"{config.savedir}/summary.txt", "w"),
        )
