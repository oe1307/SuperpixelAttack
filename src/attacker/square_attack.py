import torch
from art.attacks.evasion import SquareAttack
from art.estimators.classification import PyTorchClassifier
from torch import Tensor

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class SquareAttack(Attacker):
    def __init__(self):
        super().__init__()

    def recorder(self):
        super().recorder()
        self.best_loss = torch.zeros(
            (config.n_examples, 2),
            dtype=torch.float16,
            device=config.device,
        )
        self.current_loss = torch.zeros(
            (config.n_examples, 2),
            dtype=torch.float16,
            device=config.device,
        )

    def _attack(self, x: Tensor, y: Tensor):
        breakpoint()
        model = PyTorchClassifier(self.model, self.criterion, nb_classes=10)
        attack = SquareAttack()
