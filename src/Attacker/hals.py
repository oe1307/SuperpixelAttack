import heapq
import math
from typing import Union

import torch
from torch import Tensor

from Base import Attacker, get_criterion
from Utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class HALS(Attacker):
    def __init__(self):
        super().__init__()
        if (config.dataset == "cifar10" and config.initial_split != 4) or (
            config.dataset == "imagenet" and config.initial_split != 32
        ):
            logger.warning(f"{config.dataset}: split = {config.initial_split}")
        self.criterion = get_criterion()
        self.num_forward = config.steps

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        x_adv_all = []
        for idx, (x, y) in enumerate(zip(x_all, y_all)):
            pbar(idx + 1, x_all.shape[0])

            # initialize
            self.upper = (x + config.epsilon).clamp(0, 1).clone()
            self.lower = (x - config.epsilon).clamp(0, 1).clone()
            self.split = config.initial_split
            c, h, w = x.shape
            is_upper = torch.zeros(
                (c, h // self.split, w // self.split),
                dtype=torch.bool,
                device=config.device,
            )
            x_adv = self.lower.unsqueeze(0).clone()
            loss = self.criterion(self.model(x_adv), y).clone()
            self.forward = 1

            while True:
                is_upper, loss = self.local_search(is_upper, loss, y)
                if self.forward >= self.num_forward:
                    break
                elif self.split > 1:
                    is_upper = is_upper.repeat(1, 2, 2)
                    self.split //= 2

            _is_upper = is_upper.repeat(1, self.split, self.split)
            x_adv = torch.where(_is_upper, self.upper, self.lower)
            x_adv_all.append(x_adv)
        x_adv_all = torch.stack(x_adv_all)
        return x_adv_all

    def local_search(
        self, is_upper: Tensor, loss: Tensor, y: Tensor
    ) -> Union[Tensor, Tensor]:
        for iter in range(config.local_search_iteration):

            if self.forward >= self.num_forward:
                break
            is_upper, loss = self.insert(is_upper, loss, y)

            if self.forward >= self.num_forward:
                break
            is_upper, loss = self.deletion(is_upper, loss, y)

        # argmax {S, S \ V}
        if self.forward < self.num_forward:
            _is_upper = is_upper.repeat(1, self.split, self.split)
            x_adv_inverse = torch.where(_is_upper, self.lower, self.upper).unsqueeze(0)
            loss_inverse = self.criterion(self.model(x_adv_inverse), y)
            if loss_inverse > loss:
                is_upper = ~is_upper
                loss = loss_inverse
        return is_upper, loss

    def insert(
        self, is_upper: Tensor, base_loss: Tensor, y: Tensor
    ) -> Union[Tensor, Tensor]:
        max_heap = []
        all_elements = (~is_upper).nonzero()

        # search in elementary
        num_batch = math.ceil(all_elements.shape[0] / self.model.batch_size)
        for batch in range(num_batch):
            if self.forward >= self.num_forward:
                break
            start = batch * self.model.batch_size
            end = min((batch + 1) * self.model.batch_size, all_elements.shape[0])
            elements = all_elements[start:end]
            _is_upper = is_upper.repeat(elements.shape[0], 1, 1, 1)
            for i, (c, h, w) in enumerate(elements):
                assert _is_upper[i, c, h, w].item() is False
                _is_upper[i, c, h, w] = True
                self.forward += 1
                if self.forward >= self.num_forward:
                    break
            _is_upper = _is_upper.repeat(1, 1, self.split, self.split)
            x_adv = torch.where(_is_upper, self.upper, self.lower)
            loss = self.criterion(self.model(x_adv), y).clone()
            for i, (c, h, w) in enumerate(elements.tolist()):
                delta = (base_loss - loss[i]).item()
                heapq.heappush(max_heap, (delta, (c, h, w)))

        # update
        for delta, (c, h, w) in max_heap:
            while len(max_heap) > 1:
                delta_hat, element_hat = heapq.heappop(max_heap)
                delta_tilde = max_heap[0][0]
                if delta_hat <= delta_tilde and delta_hat < 0:
                    assert is_upper[element_hat].item() is False
                    is_upper[element_hat] = True
                elif delta_hat <= delta_tilde and delta_hat >= 0:
                    break
                else:
                    heapq.heappush(max_heap, (delta_hat, element_hat))
        _is_upper = is_upper.repeat([1, self.split, self.split])
        x_adv = torch.where(_is_upper, self.upper, self.lower).unsqueeze(0)
        loss = self.criterion(self.model(x_adv), y).clone()
        return is_upper, loss

    def deletion(
        self, is_upper: Tensor, base_loss: Tensor, y: Tensor
    ) -> Union[Tensor, Tensor]:
        max_heap = []
        all_elements = is_upper.nonzero()

        # search in elementary
        num_batch = math.ceil(all_elements.shape[0] / self.model.batch_size)
        for batch in range(num_batch):
            if self.forward >= self.num_forward:
                break
            start = batch * self.model.batch_size
            end = min((batch + 1) * self.model.batch_size, all_elements.shape[0])
            elements = all_elements[start:end]
            _is_upper = is_upper.repeat(elements.shape[0], 1, 1, 1)
            for i, (c, h, w) in enumerate(elements):
                assert _is_upper[i, c, h, w].item() is True
                _is_upper[i, c, h, w] = False
                self.forward += 1
                if self.forward >= self.num_forward:
                    break
            _is_upper = _is_upper.repeat(1, 1, self.split, self.split)
            x_adv = torch.where(_is_upper, self.upper, self.lower)
            loss = self.criterion(self.model(x_adv), y).clone()
            for i, (c, h, w) in enumerate(elements.tolist()):
                delta = (base_loss - loss[i]).item()
                heapq.heappush(max_heap, (delta, (c, h, w)))

        # update
        for delta, (c, h, w) in max_heap:
            while len(max_heap) > 1:
                delta_hat, element_hat = heapq.heappop(max_heap)
                delta_tilde = max_heap[0][0]
                if delta_hat <= delta_tilde and delta_hat < 0:
                    assert is_upper[element_hat].item() is True
                    is_upper[element_hat] = False
                elif delta_hat <= delta_tilde and delta_hat >= 0:
                    break
                else:
                    heapq.heappush(max_heap, (delta_hat, element_hat))
        _is_upper = is_upper.repeat([1, self.split, self.split])
        x_adv = torch.where(_is_upper, self.upper, self.lower).unsqueeze(0)
        loss = self.criterion(self.model(x_adv), y).clone()
        return is_upper, loss
