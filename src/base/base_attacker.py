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
        self.timekeeper = time.time()
        config.savedir = rename_dir(f"../result/{config.attacker}")
        config_parser.save(f"{config.savedir}/config.json")
        shutil.copytree("../src", f"{config.savedir}/backup")

        # remove misclassification images
        clean_acc = torch.zeros(config.n_examples, device=config.device, dtype=bool)
        n_batch = math.ceil(config.n_examples / config.batch_size)
        for i in range(n_batch):
            start = i * config.batch_size
            end = min((i + 1) * config.batch_size, config.n_examples)
            x = x_all[start:end]
            y = y_all[start:end]
            pred = self.model(x).softmax(dim=1)
            clean_acc[start:end] = pred.argmax(dim=1) == y
        config.clean_acc = clean_acc.sum().item()

        x_adv = self._attack(x_all[clean_acc], y_all[clean_acc])
        x_adv_all = x_all.clone()
        x_adv_all[clean_acc] = x_adv
        torch.cuda.empty_cache()

        # calculate robust accuracy
        assert x_adv_all.shape == x_all.shape
        robust_acc = torch.zeros(config.n_examples, device=config.device, dtype=bool)
        for i in range(n_batch):
            start = i * config.batch_size
            end = min((i + 1) * config.batch_size, config.n_examples)
            x_clean = x_all[start:end]
            x_adv = x_adv_all[start:end]
            y = y_all[start:end]

            # for check
            upper = (x_clean + config.epsilon).clamp(0, 1).clone()
            lower = (x_clean - config.epsilon).clamp(0, 1).clone()
            assert (x_adv <= upper + 1e-10).all() and (x_adv >= lower - 1e-10).all()
            x_adv = x_adv.clamp(lower, upper)

            pred = self.model(x_adv).softmax(dim=1)
            robust_acc[start:end] = pred.argmax(dim=1) == y
            np.save(f"{config.savedir}/robust_acc.npy", robust_acc.cpu().numpy())
        total_time = time.time() - self.timekeeper
        attack_success_rate = 100 - robust_acc.sum() / config.n_examples * 100

        msg = (
            "\n"
            + f"n_examples = {config.n_examples}\n"
            + f"model = {self.model.name}\n"
            + f"epsilon = {config.epsilon}\n"
            + f"forward = {config.n_forward}\n"
            + f"total time = {total_time:.2f} s\n"
            + f"attack success rate = {attack_success_rate:.2f} %\n"
        )
        print(msg, file=open(f"{config.savedir}/summary.txt", "w"))
        logger.info(msg)
        config_parser.save(f"{config.savedir}/config.json")
