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
        x_adv = x.clone()

        for _ in range(config.restart):
            for iter in range(config.iteration):
                step_size = self.step_size_manager(iter, x[0].numel())
                # self.robust_acc(x_adv, y)

        return x_adv

    def step_size_manager(self, iter: int, total: int) -> float:
        """Step size manager for Tabu Attack.
        Args:
            iter (int): Current iteration.
            total (int): Total number of pixels.
        Returns:
            step_size (float): Step size for current iteration.
        Note:
            step size means L1 norm
        """
        if config.step_size == "elementally":
            step_size = 1
        elif config.step_size == "linear":
            step_size = int(total * (1 - iter / config.iteration))

        logger.debug(f"step size: {step_size}")
        return step_size
