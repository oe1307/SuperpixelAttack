import heapq
import math

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
        assert config.iteration % 2 == 0, "iteration should be even"
        upper = (x + config.epsilon).clamp(0, 1).detach().clone()
        lower = (x - config.epsilon).clamp(0, 1).detach().clone()

        # initialize
        split = config.initial_split
        is_upper = torch.zeros(
            (x.shape[0], x.shape[1], x.shape[2] // split, x.shape[3] // split),
            dtype=torch.bool,
            device=config.device,
        )
        logger.debug(f"\nmask shape: {list(is_upper.shape)}")
        x_adv = lower.detach().clone()
        loss = self.robust_acc(x_adv, y).detach().clone()

        # repeat
        for _ in range(config.iteration // 2 - 1):
            logger.debug(f"\nmask shape: {list(is_upper.shape)}")
            is_upper = self.local_search(upper, lower, is_upper, split, y, loss)
            if split > 1:
                # split block
                is_upper = is_upper.repeat([1, 1, 2, 2])
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
        for iter in range(config.max_iter):
            logger.debug(f"insert ( iter={iter} )")
            is_upper = self.insert(is_upper, y, upper, lower, split, loss)
            logger.debug(f"deletion ( iter={iter} )")
            is_upper = self.deletion(is_upper, y, upper, lower, split, loss)
        _is_upper = is_upper.repeat([1, 1, split, split]).to(config.device)

        x_adv_1 = upper * _is_upper + lower * ~_is_upper
        loss_1 = self.robust_acc(x_adv_1, y)

        x_adv_2 = upper * ~_is_upper + lower * _is_upper
        loss_2 = self.robust_acc(x_adv_2, y)

        return torch.where((loss_2 < loss_1).view(-1, 1, 1, 1), is_upper, ~is_upper)

    @torch.inference_mode()
    def insert(
        self, is_upper: Tensor, y: Tensor, upper, lower, split, base_loss: Tensor
    ) -> Tensor:
        max_heap = [[] for _ in range(self.end - self.start)]
        all_elements = (~is_upper).nonzero()

        # search in elementary
        num_batch = math.ceil(all_elements.shape[0] / self.model.batch_size)
        for i in range(num_batch):
            _start = i * self.model.batch_size
            _end = min((i + 1) * self.model.batch_size, all_elements.shape[0])
            elements = all_elements[_start:_end]
            _is_upper = is_upper[elements[:, 0]]
            for i, (c, h, w) in enumerate(elements[:, 1:]):
                _is_upper[i, c, h, w] = True
            _is_upper = _is_upper.repeat([1, 1, split, split])
            x_adv = (
                upper[elements[:, 0]] * _is_upper + lower[elements[:, 0]] * ~_is_upper
            )
            loss = self.criterion(self.model(x_adv), y[elements[:, 0]])
            self.num_forward += _end - _start
            for i, (idx, c, h, w) in enumerate(elements.tolist()):
                delta = (base_loss[idx] - loss[i]).item()
                heapq.heappush(max_heap[idx], (delta, (c, h, w)))

        # update
        _is_upper = []
        for idx, _max_heap in enumerate(max_heap):
            idx_is_upper = is_upper[idx]
            while len(_max_heap) > 1:
                delta_hat, element_hat = heapq.heappop(_max_heap)
                delta_tilde = _max_heap[0][0]
                if delta_hat <= delta_tilde and delta_hat < 0:
                    idx_is_upper[element_hat] = True
                elif delta_hat <= delta_tilde and delta_hat >= 0:
                    break
                else:
                    heapq.heappush(_max_heap, (delta_hat, element_hat))
            _is_upper.append(idx_is_upper)
        return torch.stack(_is_upper)

    @torch.inference_mode()
    def deletion(
        self, is_upper: Tensor, y: Tensor, upper, lower, split, base_loss: Tensor
    ) -> Tensor:
        max_heap = [[] for _ in range(self.end - self.start)]
        all_elements = is_upper.nonzero()

        # search in elementary
        num_batch = math.ceil(all_elements.shape[0] / self.model.batch_size)
        for i in range(num_batch):
            _start = i * self.model.batch_size
            _end = min((i + 1) * self.model.batch_size, all_elements.shape[0])
            elements = all_elements[_start:_end]
            _is_upper = is_upper[elements[:, 0]]
            for i, (c, h, w) in enumerate(elements[:, 1:]):
                _is_upper[i, c, h, w] = False
            _is_upper = _is_upper.repeat([1, 1, split, split])
            x_adv = (
                upper[elements[:, 0]] * _is_upper + lower[elements[:, 0]] * ~_is_upper
            )
            loss = self.criterion(self.model(x_adv), y[elements[:, 0]])
            self.num_forward += _end - _start
            for i, (idx, c, h, w) in enumerate(elements.tolist()):
                delta = (base_loss[idx] - loss[i]).item()
                heapq.heappush(max_heap[idx], (delta, (c, h, w)))

        # update
        _is_upper = []
        for idx, _max_heap in enumerate(max_heap):
            idx_is_upper = is_upper[idx]
            while len(_max_heap) > 1:
                delta_hat, element_hat = heapq.heappop(_max_heap)
                delta_tilde = _max_heap[0][0]
                if delta_hat <= delta_tilde and delta_hat < 0:
                    idx_is_upper[element_hat] = False
                elif delta_hat <= delta_tilde and delta_hat >= 0:
                    break
                else:
                    heapq.heappush(_max_heap, (delta_hat, element_hat))
            _is_upper.append(idx_is_upper)
        return torch.stack(_is_upper)
