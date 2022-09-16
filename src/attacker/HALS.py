import heapq
import math

import torch
from torch import Tensor

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class HALS_Attacker(Attacker):
    """Hierarchical Accelerated Local Search"""

    def __init__(self):
        super().__init__()

    def _recorder(self):
        self.best_loss = torch.zeros(
            (config.n_examples, 3 * config.iteration + 2),
            dtype=torch.float16,
            device=config.device,
        )
        self.current_loss = torch.zeros(
            (config.n_examples, 3 * config.iteration + 2),
            dtype=torch.float16,
            device=config.device,
        )

    @torch.inference_mode()
    def _attack(self, x: Tensor, y: Tensor):
        upper = (x + config.epsilon).clamp(0, 1).detach().clone()
        lower = (x - config.epsilon).clamp(0, 1).detach().clone()

        # initialize
        split = config.initial_split
        batch, c, h, w = x.shape
        is_upper = torch.zeros(
            (batch, c, h // split, w // split), dtype=torch.bool, device=config.device
        )
        logger.debug(f"\nmask shape: {[c, h // split, w // split]}")
        x_adv = lower.detach().clone()
        loss = self.robust_acc(x_adv, y).detach().clone()

        # repeat
        for _ in range(config.iteration):
            logger.debug(f"\nmask shape: {[c, h // split, w // split]}")
            is_upper, loss = self.local_search(upper, lower, is_upper, split, y, loss)
            if split > 1:
                # split block
                is_upper = is_upper.repeat([1, 1, 2, 2])
                if split % 2 == 1:
                    logger.warning(f"split is not even: {split}")
                split //= 2

        _is_upper = is_upper.repeat([1, 1, split, split])
        x_adv = torch.where(_is_upper, upper, lower).detach().clone()
        return x_adv

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
        for _ in range(config.max_iter):
            is_upper, loss = self.insert(is_upper, y, upper, lower, split, loss)
            is_upper, loss = self.deletion(is_upper, y, upper, lower, split, loss)
        _is_upper = is_upper.repeat([1, 1, split, split])
        x_adv_inverse = torch.where(~_is_upper, upper, lower).detach().clone()
        loss_inverse = self.robust_acc(x_adv_inverse, y).detach().clone()
        is_upper = torch.where(
            (loss_inverse < loss).view(-1, 1, 1, 1), is_upper, ~is_upper
        )
        loss = torch.max(loss, loss_inverse)
        return is_upper, loss

    @torch.inference_mode()
    def insert(
        self,
        is_upper: Tensor,
        y: Tensor,
        upper: Tensor,
        lower: Tensor,
        split: int,
        base_loss: Tensor,
    ) -> Tensor:
        max_heap = [[] for _ in range(self.end - self.start)]
        all_elements = (~is_upper).nonzero()

        # search in elementary
        num_batch = math.ceil(all_elements.shape[0] / self.model.batch_size)
        for i in range(num_batch):
            _start = i * self.model.batch_size
            _end = min((i + 1) * self.model.batch_size, all_elements.shape[0])
            elements = all_elements[_start:_end]
            _is_upper = is_upper[elements[:, 0]].clone()
            for i, (c, h, w) in enumerate(elements[:, 1:]):
                assert _is_upper[i, c, h, w] is False
                _is_upper[i, c, h, w] = True
            _is_upper = _is_upper.repeat([1, 1, split, split])
            x_adv = torch.where(_is_upper, upper[elements[:, 0]], lower[elements[:, 0]])
            loss = self.criterion(self.model(x_adv), y[elements[:, 0]]).detach().clone()
            self.num_forward += x_adv.shape[0]
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
                    assert idx_is_upper[element_hat] is False
                    idx_is_upper[element_hat] = True
                elif delta_hat <= delta_tilde and delta_hat >= 0:
                    break
                else:
                    heapq.heappush(_max_heap, (delta_hat, element_hat))
            _is_upper.append(idx_is_upper)
        is_upper = torch.stack(_is_upper)
        _is_upper = is_upper.repeat([1, 1, split, split])
        x_adv = torch.where(_is_upper, upper, lower).detach().clone()
        loss = self.robust_acc(x_adv, y).detach().clone()
        return is_upper, loss

    @torch.inference_mode()
    def deletion(
        self,
        is_upper: Tensor,
        y: Tensor,
        upper: Tensor,
        lower: Tensor,
        split: int,
        base_loss: Tensor,
    ) -> Tensor:
        max_heap = [[] for _ in range(self.end - self.start)]
        all_elements = is_upper.nonzero()

        # search in elementary
        num_batch = math.ceil(all_elements.shape[0] / self.model.batch_size)
        for i in range(num_batch):
            _start = i * self.model.batch_size
            _end = min((i + 1) * self.model.batch_size, all_elements.shape[0])
            elements = all_elements[_start:_end]
            _is_upper = is_upper[elements[:, 0]].clone()
            for i, (c, h, w) in enumerate(elements[:, 1:]):
                assert _is_upper[i, c, h, w] is True
                _is_upper[i, c, h, w] = False
            _is_upper = _is_upper.repeat([1, 1, split, split])
            x_adv = torch.where(_is_upper, upper[elements[:, 0]], lower[elements[:, 0]])
            loss = self.criterion(self.model(x_adv), y[elements[:, 0]]).detach().clone()
            self.num_forward += x_adv.shape[0]
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
                    assert idx_is_upper[element_hat] is True
                    idx_is_upper[element_hat] = False
                elif delta_hat <= delta_tilde and delta_hat >= 0:
                    break
                else:
                    heapq.heappush(_max_heap, (delta_hat, element_hat))
            _is_upper.append(idx_is_upper)
        is_upper = torch.stack(_is_upper)
        _is_upper = is_upper.repeat([1, 1, split, split])
        x_adv = torch.where(_is_upper, upper, lower).detach().clone()
        loss = self.robust_acc(x_adv, y).detach().clone()
        return is_upper, loss
