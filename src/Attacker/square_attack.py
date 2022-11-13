import math

import art
import numpy as np
import torch
from art.estimators.classification import ClassifierMixin, PyTorchClassifier
from torch import Tensor
from yaspin import yaspin

from base import Attacker
from utils import change_level, config_parser, setup_logger, pbar

logger = setup_logger(__name__)
config = config_parser()


class SquareAttack(Attacker):
    def __init__(self):
        config.n_forward = config.restart * config.steps

    def _attack(self, x: Tensor, y: Tensor) -> Tensor:
        change_level("art", 40)
        torch.cuda.current_device = lambda: config.device
        model = PyTorchClassifier(
            self.model,
            ClassifierMixin,
            input_shape=x.shape[1:],
            nb_classes=config.n_classes,
            clip_values=(0, 1),
        )
        model._device = torch.device(config.device)
        attack = art.attacks.evasion.SquareAttack(
            estimator=model,
            norm=np.inf,
            max_iter=config.steps,
            eps=config.epsilon,
            p_init=config.p_init,
            nb_restarts=config.restart,
            batch_size=model.batch_size,
            verbose=False,
        )
        x_adv = attack.generate(x.cpu().numpy())

        upper = (x + config.epsilon).clamp(0, 1).clone()
        lower = (x - config.epsilon).clamp(0, 1).clone()
        assert torch.all(x_adv <= upper + 1e-10)
        assert torch.all(x_adv >= lower - 1e-10)
        logits = self.model(x_adv).clone()
        self.robust_acc = logits.argmax(dim=1) == y
