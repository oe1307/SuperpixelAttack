import torch

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class PGD_Attacker(Attacker):
    """Project Gradient Descent Attack"""

    def __init__(self):
        super().__init__()

    def _attack(self, x, y):
        upper = (x + config.epsilon).clamp(0, 1).detach().clone()
        lower = (x - config.epsilon).clamp(0, 1).detach().clone()
        x_adv = x.clone().requires_grad_()

        loss = self.criterion(self.model(x_adv), y).sum()
        self.num_forward += x_adv.shape[0]
        for _ in range(config.iteration):
            grad = torch.autograd.grad(loss, [x_adv])[0]
            self.num_backward += x_adv.shape[0]
            x_adv = (x_adv + config.step_size * torch.sign(grad)).clamp(lower, upper)
            del grad
            assert torch.all(x_adv <= upper + 1e-6) and torch.all(x_adv >= lower - 1e-6)
            loss = self.robust_acc(x_adv, y).sum()
