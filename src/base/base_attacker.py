import os
import time

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from utils import config_parser, rename_dir, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class Attacker:
    @torch.no_grad()
    def attack(self, model: Module, x: Tensor, y: Tensor) -> Tensor:
        """
        model (Module): classifier
        x (Tensor): clean image
        y (Tensor): correct label
        """
        assert not model.training
        self.model = model
        x = x.to(config.device)
        y = y.to(config.device)
        self.timekeeper = time.time()
        self.robust_acc = torch.zeros(x.shape[0], dtype=torch.bool)
        config.savedir = rename_dir(f"../result/{config.dataset}/{config.attacker}")
        os.makedirs(config.savedir, exist_ok=True)
        config_parser.save(f"{config.savedir}/config.json")

        x_adv = self._attack(x, y)

        assert x_adv.shape == x.shape
        upper = (x + config.epsilon).clamp(0, 1).clone()
        lower = (x - config.epsilon).clamp(0, 1).clone()
        assert (x_adv <= upper + 1e-10).all() and (x_adv >= lower - 1e-10).all()
        np.save(f"{config.savedir}/x_adv.npy", x_adv.cpu().numpy())
        logit = self.model(x_adv).clone()
        self.robust_acc = logit.argmax(dim=1) == y
        np.save(f"{config.savedir}/robust_acc.npy", self.robust_acc.cpu().numpy())
        total_time = time.time() - self.timekeeper
        robust_acc = self.robust_acc.sum() / x.shape[0] * 100
        ASR = 100 - robust_acc

        msg = (
            "\n"
            + f"n_img = {x.shape[0]}\n"
            + f"total time (sec) = {total_time:.2f}s\n"
            + f"ASR (%) = {ASR:.2f}\n"
        )
        print(msg, file=open(f"{config.savedir}/summary.txt", "w"))
        logger.info(msg)
