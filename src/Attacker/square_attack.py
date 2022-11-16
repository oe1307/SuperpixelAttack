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

    def _attack(self, x: Tensor, y: Tensor) -> Tensor:
        clean_acc = torch.zeros_like(y, dtype=torch.bool)
        n_batch = math.ceil(x.shape[0] / self.model.batch_size)
        for i in range(n_batch):
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, config.n_examples)
            _x = x[start:end].to(config.device)
            _y = y[start:end].to(config.device)
            logits = self.model(_x).clone()
            clean_acc[start:end] = logits.argmax(dim=1) == _y

        change_level("art", 40)
        torch.cuda.current_device = lambda: config.device
        model = PyTorchClassifier(
            self.model,
            loss=None,
            input_shape=x.shape[1:],
            nb_classes=config.n_classes,
            clip_values=(0, 1),
        )
        model._device = torch.device(config.device)

        attack = art.attacks.evasion.SquareAttack(
            estimator=model,
            eps=config.epsilon,
            batch_size=self.model.batch_size,
            verbose=False,
            # hyperparameter
            max_iter=config.steps,
            p_init=config.p_init,
            nb_restarts=config.restart,
        )
        with yaspin(text="Attacking...", color="cyan"):
            x_adv = attack.generate(x.cpu().numpy())
        x_adv = torch.from_numpy(x_adv).to(config.device)
        x_adv = torch.where(clean_acc.view(-1, 1, 1, 1), x, x_adv)
        return x_adv
