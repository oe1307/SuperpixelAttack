import torch
from torch import Tensor

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class TabuAttack(Attacker):
    def __init__(self):
        super().__init__()

    def recorder(self):
        super().recorder()
        self.best_loss = torch.zeros(
            (config.n_examples, config.restart * config.iteration + 1),
            dtype=torch.float16,
            device=config.device,
        )
        self.current_loss = torch.zeros(
            (config.n_examples, config.restart * config.iteration + 1),
            dtype=torch.float16,
            device=config.device,
        )

    @torch.inference_mode()
    def _attack(self, x: Tensor, y: Tensor):
        upper = (x + config.epsilon).clamp(0, 1).clone()
        lower = (x - config.epsilon).clamp(0, 1).clone()
        tabu_list = torch.zeros_like(x, dtype=torch.uint8)

        is_upper_best = torch.randint_like(x, 0, 2, dtype=torch.bool)
        x_best = torch.where(is_upper_best, upper, lower).clone()
        self.robust_acc(x_best, y)
        is_upper = is_upper_best.clone()
        for iter in range(config.iteration):
            delta = torch.randint(0, x[0].numel(), (x.shape[0],))
            is_upper = is_upper.view(x.shape[0], -1)
            is_upper[torch.arange(x.shape[0]), delta] = ~is_upper[
                torch.arange(x.shape[0]), delta
            ]
            is_upper = is_upper.view(x.shape)
            x_adv = torch.where(is_upper, upper, lower).clone()
            self.robust_acc(x_adv, y)
            updated = self.check(iter)
            x_best[updated] = x_adv[updated]
            breakpoint()

        assert torch.all(lower <= x_best <= upper)
        return x_best

    def check(self, iter):
        current_loss = self.current_loss[self.start : self.end, iter + 2]
        best_loss = self.best_loss[self.start : self.end, iter + 2]
        return best_loss == current_loss
