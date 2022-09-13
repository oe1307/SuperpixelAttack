import heapq

import torch
from torch import Tensor

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)


class HALS_Attacker(Attacker):
    """Hierarchical Accelerated Local Search"""

    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def _attack(self, x: Tensor, y: Tensor):
        config = config_parser.config
        upper = (x + config.epsilon).clamp(0, 1).detach().clone()
        lower = (x - config.epsilon).clamp(0, 1).detach().clone()

        # initialize
        split = config.initial_split
        is_upper = torch.zeros(
            (x.shape[0], x.shape[1], x.shape[2] // split, x.shape[3] // split),
            dtype=torch.bool,
            device=config.device,
        )
        logger.debug(f"mask shape: {list(is_upper.shape)}")
        x_adv = lower.detach().clone()
        loss = self.robust_acc(x_adv, y).detach().clone()

        # repeat
        for _ in range(config.num_search):
            is_upper = self.local_search(upper, lower, is_upper, split, y, loss)
            if split > 1:
                # split block
                is_upper = is_upper.repeat([1, 1, 2, 2])
                logger.debug(f"mask shape: {list(is_upper.shape)}")
                split //= 2

    @torch.inference_mode()
    def local_search(
        self,
        upper: Tensor,
        lower: Tensor,
        is_upper: Tensor,
        split: int,
        y: Tensor,
        loss: Tensor,
    ):
        config = config_parser.config
        for _ in range(config.iteration):
            is_upper = self.insert(is_upper, y, upper, lower, split, loss)
            breakpoint()
            is_upper = self.deletion(is_upper, y, upper, lower, split, loss)
        _is_upper = is_upper.repeat([1, 1, split, split]).to(config.device)

        x_adv_1 = upper * _is_upper + lower * ~_is_upper
        loss_1 = self.robust_acc(x_adv_1, y)

        x_adv_2 = upper * ~_is_upper + lower * _is_upper
        loss_2 = self.robust_acc(x_adv_2, y)

        return is_upper if loss_2 < loss_1 else ~is_upper

    @torch.inference_mode()
    def insert(
        self, is_upper: Tensor, y: Tensor, upper, lower, split, base_loss: Tensor
    ) -> Tensor:
        _is_upper = []
        for idx in range(is_upper.shape[0]):  # FIXME: batch
            max_heap = []
            for c, h, w in (~is_upper[idx]).nonzero().tolist():
                idx_is_upper = is_upper[idx].detach().clone()
                idx_is_upper[c, h, w] = not idx_is_upper[c, h, w]
                _idx_is_upper = idx_is_upper.repeat([1, split, split])
                x_adv = upper[idx] * _idx_is_upper + lower[idx] * ~_idx_is_upper
                loss = self.idx_robust_acc(x_adv, y, self.start + idx).detach().clone()
                delta = (base_loss[idx] - loss).item()
                heapq.heappush(max_heap, (delta, [c, h, w]))
            while len(max_heap) > 0:
                delta, (c, h, w) = heapq.heappop(max_heap)
                self.current_loss[idx, self.iter] = base_loss[idx] - delta
                self.best_loss[idx, self.iter] = torch.max(
                    base_loss[idx] - delta, self.best_loss[idx, self.iter - 1]
                )
                idx_is_upper = is_upper[idx].detach().clone()
                idx_is_upper[c, h, w] = not idx_is_upper[c, h, w]
            _is_upper.append(idx_is_upper)
        return torch.stack(_is_upper)

    @torch.inference_mode()
    def deletion(self, is_upper: Tensor) -> Tensor:
        return is_upper
