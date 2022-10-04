import torch

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class PGD(Attacker):
    """Project Gradient Descent Attack"""

    def __init__(self):
        super().__init__()

    def recorder(self):
        super().recorder()
        self.best_loss = torch.zeros(
            (config.n_examples, config.iteration + 1),
            dtype=torch.float16,
            device=config.device,
        )
        self.current_loss = torch.zeros(
            (config.n_examples, config.iteration + 1),
            dtype=torch.float16,
            device=config.device,
        )

    def _attack(self, x, y):
        upper = (x + config.epsilon).clamp(0, 1).clone()
        lower = (x - config.epsilon).clamp(0, 1).clone()
        x_adv = x.clone().requires_grad_()

        loss = self.criterion(self.model(x_adv), y).sum().clone()
        self.num_forward += x_adv.shape[0]
        for _ in range(config.iteration):
            grad = torch.autograd.grad(loss, [x_adv])[0].clone()
            self.num_backward += x_adv.shape[0]
            x_adv = (
                (x_adv + config.step_size * torch.sign(grad))
                .clamp(lower, upper)
                .clone()
            )
            del grad
            assert torch.all(x_adv <= upper + 1e-6) and torch.all(x_adv >= lower - 1e-6)
            loss = self.robust_acc(x_adv, y).sum().clone()

    def record(self):
        super().record()
        self.num_forward = (
            self.num_forward
            * self.success_iter.sum()
            / (config.n_examples * (config.iteration + 1))
        ).to(torch.int16)
        self.num_backward = (
            self.num_backward
            * self.success_iter.sum()
            / (config.n_examples * (config.iteration + 1))
        ).to(torch.int16)
        msg = (
            f"num_forward = {self.num_forward}\n"
            + f"num_backward = {self.num_backward}"
        )
        print(msg, file=open(config.savedir + "/summary.txt", "a"))
        logger.info(msg + "\n")
