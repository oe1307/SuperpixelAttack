import math

import art
import torch
from art.estimators.classification import PyTorchClassifier
from torch import Tensor
from yaspin import yaspin

from base import Attacker
from utils import change_level, config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class SquareAttack(Attacker):
    def __init__(self):
        config.n_forward = config.restart * config.steps

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        # 元々誤分類の画像を排除
        n_images = x_all.shape[0]
        clean_acc = torch.zeros_like(y_all, dtype=torch.bool)
        n_batch = math.ceil(n_images / self.model.batch_size)
        for i in range(n_batch):
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, config.n_examples)
            x = x_all[start:end].to(config.device)
            y = y_all[start:end].to(config.device)
            logits = self.model(x).clone()
            clean_acc[start:end] = logits.argmax(dim=1) == y

        change_level("art", 40)
        torch.cuda.current_device = lambda: config.device
        model = PyTorchClassifier(
            self.model,
            loss=None,
            input_shape=x_all.shape[1:],
            nb_classes=config.n_classes,
            clip_values=(0, 1),
        )
        model._device = torch.device(config.device)

        attack = art.attacks.evasion.SquareAttack(
            estimator=model,
            eps=config.epsilon,
            batch_size=self.model.batch_size,
            verbose=False,
            # ---hyperparameter---
            max_iter=config.steps,
            p_init=config.p_init,
            nb_restarts=config.restart,
        )
        assert x_all.device() == torch.device("cpu")
        with yaspin(text="Attacking...", color="cyan"):
            x_adv = attack.generate(x_all[clean_acc].numpy())
        x_adv_all = x_all.clone()
        x_adv_all[clean_acc] = torch.from_numpy(x_adv).to(config.device)
        return x_adv
