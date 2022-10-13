import numpy as np
import torch
from torch import Tensor

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class LocalSearch(Attacker):
    """method3 との比較用"""

    def __init__(self):
        super().__init__()

    def recorder(self):
        super().recorder()
        self.best_loss = torch.zeros(
            (config.n_examples, config.iteration), device=config.device
        )
        self.current_loss = torch.zeros(
            (config.n_examples, config.iteration), device=config.device
        )

    @torch.inference_mode()
    def _attack(self, x_all: Tensor, y_all: Tensor):
        x_adv_all = []
        for self.idx, (x, y) in enumerate(zip(x_all, y_all)):
            # initialize
            upper = (x + config.epsilon).clamp(0, 1).clone()
            lower = (x - config.epsilon).clamp(0, 1).clone()
            _is_upper = torch.randint_like(x, 0, 2, dtype=torch.bool)
            x_best = torch.where(_is_upper, upper, lower)
            best_loss = self.robust_acc(x_best, y).item()
            self.current_loss[self.idx, 1] = best_loss
            self.best_loss[self.idx, 1] = best_loss

            for iter in range(2, config.iteration):
                flips = np.arange(x.numel())
                np.random.shuffle(flips)
                for flip in flips:
                    is_upper = _is_upper.clone()
                    is_upper.view(-1)[flip] = ~is_upper.view(-1)[flip]
                    x_adv = torch.where(is_upper, upper, lower).clone()
                    loss = self.robust_acc(x_adv, y).item()
                    logger.debug(
                        f"( iter={iter} ) loss={loss:.4f} best_loss={best_loss:.4f}"
                    )
                    if loss > best_loss:
                        break
                else:
                    break

                # end for
                _is_upper = is_upper.clone()
                best_loss = loss
                x_best = x_adv.clone()
                self.best_loss[self.idx, iter] = best_loss

                if not config.exp and loss > 0:
                    logger.info(f"idx={self.idx} iter={iter} success")
                    break

            assert torch.all(x_adv <= upper + 1e-6) and torch.all(x_adv >= lower - 1e-6)
            x_adv_all.append(x_best)
        x_adv_all = torch.stack(x_adv_all)
        return x_adv_all

    def robust_acc(self, x_adv: Tensor, y: Tensor) -> Tensor:
        """index-wise robust accuracy"""
        assert x_adv.dim() == 3
        x_adv = x_adv.unsqueeze(0)
        logits = self.model(x_adv).clone()
        self.num_forward += 1
        self._robust_acc[self.idx] = torch.logical_and(
            self._robust_acc[self.idx], logits.argmax(dim=1) == y
        )
        self.success_iter[self.idx] += self._robust_acc[self.idx]
        loss = self.criterion(logits, y).clone()
        return loss
