from art.attacks.evasion import SquareAttack
from art.estimators.classification import PyTorchClassifier
from torch import Tensor

from base import Attacker


class SquareAttack(Attacker):
    def __init__(self):
        super().__init__()

    def _attack(self, x: Tensor, y: Tensor):
        model = PyTorchClassifier(self.model, self.criterion)
        attack = SquareAttack()
