import math
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
        n_images = x.shape[0]
        x = x.to(config.device)
        y = y.to(config.device)
        self.timekeeper = time.time()
        config.savedir = rename_dir(f"../result/{config.dataset}/{config.attacker}")
        os.makedirs(config.savedir, exist_ok=True)
        config_parser.save(f"{config.savedir}/config.json")

        x_adv_all = self._attack(x, y)

        assert x_adv_all.shape == x.shape
        np.save(f"{config.savedir}/x_adv.npy", x_adv_all.cpu().numpy())
        robust_acc = torch.zeros(n_images, dtype=torch.bool)
        n_batch = math.ceil(n_images / model.batch_size)
        for i in range(n_batch):
            start = i * model.batch_size
            end = min((i + 1) * model.batch_size, n_images)
            x_clean = x[start:end]
            x_adv = x_adv_all[start:end]
            label = y[start:end]
            upper = (x_clean + config.epsilon).clamp(0, 1).clone().to(config.device)
            lower = (x_clean - config.epsilon).clamp(0, 1).clone().to(config.device)

            # for check
            assert (x_adv <= upper + 1e-10).all() and (x_adv >= lower - 1e-10).all()
            x_adv = x_adv.clamp(lower, upper)

            logit = self.model(x_adv).clone()
            robust_acc[start:end] = logit.argmax(dim=1) == label
            np.save(f"{config.savedir}/robust_acc.npy", robust_acc.cpu().numpy())
        total_time = time.time() - self.timekeeper
        ASR = 100 - robust_acc.sum() / x.shape[0] * 100

        msg = (
            "\n"
            + f"n_img = {x.shape[0]}\n"
            + f"epsilon = {config.epsilon}\n"
            + f"forward = {config.n_forward}\n"
            + f"total time (sec) = {total_time:.2f}s\n"
            + f"ASR (%) = {ASR:.2f}\n"
        )
        print(msg, file=open(f"{config.savedir}/summary.txt", "w"))
        logger.info(msg)
