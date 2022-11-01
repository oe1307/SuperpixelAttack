import art
import numpy as np
import torch
from art.estimators.classification import ClassifierMixin, PyTorchClassifier
from torch import Tensor
from yaspin import yaspin

from Base import Attacker
from Utils import change_level, config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class SquareAttack(Attacker):
    def __init__(self):
        super().__init__()
        self.num_forward = config.restart * config.steps

    def _attack(self, x: Tensor, y: Tensor) -> Tensor:
        change_level("art", 40)
        model = PyTorchClassifier(
            self.model,
            ClassifierMixin,
            input_shape=x.shape[1:],
            nb_classes=config.num_classes,
            clip_values=(0, 1),
        )

        attack = art.attacks.evasion.SquareAttack(
            estimator=model,
            norm=np.inf,
            max_iter=config.steps,
            eps=config.epsilon,
            p_init=config.p_init,
            nb_restarts=config.restart,
            batch_size=x.shape[0],
            verbose=False,
        )
        with yaspin(text="Attacking...", color="cyan"):
            x_adv = attack.generate(x.cpu().numpy())
        x_adv = torch.from_numpy(x_adv).to(config.device)
        return x_adv