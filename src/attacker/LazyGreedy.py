import os

import torch

from base import Attacker
from utils import setup_logger

logger = setup_logger(__name__)


class LazyGreedyAttacker(Attacker):
    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def _attack(self, model, x, y):
        upper = (x + self.config.epsilon).clamp(0, 1).clone().to(self.config.device)
        lower = (x - self.config.epsilon).clamp(0, 1).clone().to(self.config.device)

        for i in range(self.config.iteration):
            pass
