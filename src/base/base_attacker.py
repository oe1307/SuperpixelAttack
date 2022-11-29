import math
import shutil
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
    def attack(self, model: Module, x_all: Tensor, y_all: Tensor) -> Tensor:
        """
        model (Module): classifier
        x_all (Tensor): clean image
        y_all (Tensor): correct label
        """
        assert not model.training
        self.model = model
        n_images = x_all.shape[0]
        self.timekeeper = time.time()
        config.savedir = rename_dir(f"../result/{config.dataset}/{config.attacker}")
        config_parser.save(f"{config.savedir}/config.json")
        shutil.copytree("../src", f"{config.savedir}/backup")

        # remove misclassification images
        clean_acc = torch.zeros(n_images, dtype=torch.bool)
        n_batch = math.ceil(n_images / self.model.batch_size)
        for i in range(n_batch):
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            y = y_all[start:end]
            logits = self.model(x).clone()
            clean_acc[start:end] = logits.argmax(dim=1) == y

        x_adv = self._attack(x_all[clean_acc], y_all[clean_acc])
        x_adv_all = x_all.clone()
        x_adv_all[clean_acc] = x_adv

        assert x_adv_all.shape == x_all.shape
        robust_acc = torch.zeros(n_images, dtype=torch.bool)
        for i in range(n_batch):
            start = i * model.batch_size
            end = min((i + 1) * model.batch_size, n_images)
            x_clean = x_all[start:end]
            x_adv = x_adv_all[start:end]
            y = y_all[start:end]
            upper = (x_clean + config.epsilon).clamp(0, 1).clone()
            lower = (x_clean - config.epsilon).clamp(0, 1).clone()

            # for check
            assert (x_adv <= upper + 1e-10).all() and (x_adv >= lower - 1e-10).all()
            x_adv = x_adv.clamp(lower, upper)

            logits = self.model(x_adv).clone()
            robust_acc[start:end] = logits.argmax(dim=1) == y
            np.save(f"{config.savedir}/robust_acc.npy", robust_acc.cpu().numpy())
        total_time = time.time() - self.timekeeper
        ASR = 100 - robust_acc.sum() / x_all.shape[0] * 100

        msg = (
            "\n"
            + f"n_img = {x.shape[0]}\n"
            + f"model = {self.model.name}\n"
            + f"epsilon = {config.epsilon}\n"
            + f"forward = {config.n_forward}\n"
            + f"total time (sec) = {total_time:.2f}s\n"
            + f"ASR (%) = {ASR:.2f}\n"
        )
        print(msg, file=open(f"{config.savedir}/summary.txt", "w"))
        logger.info(msg)
