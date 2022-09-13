import torch
from torch import Tensor

from base import Attacker
from utils import COMMENT, config_parser, setup_logger

logger = setup_logger(__name__)


class HALS_Attacker(Attacker):
    """Hierarchical Accelerated Local Search"""

    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def _attack(self, model, x, y, criterion, start, end):
        config = config_parser.config
        upper = (x + config.epsilon).clamp(0, 1)
        lower = (x - config.epsilon).clamp(0, 1)

        x_adv = lower.detach().clone()
        self._robust_acc(model, x_adv, y, criterion, start, end, iter=1)
        split = config.initial_split

        while True:
            x_adv = self.local_search()
            self._robust_acc(model, x_adv, y, criterion, start, end, iter=i + 1)
            if split > 1:
                split = self.split_block(split)
                split //= 2

        if COMMENT:
            is_upper = torch.zeros(
                (x.shape[0], x.shape[1], x.shape[2] // split, x.shape[3] // split),
                dtype=torch.bool,
            )
            _is_upper = is_upper.repeat([1, 1, split, split]).to(config.device)
            assert x.shape == _is_upper.shape
            x = upper * _is_upper + lower * ~_is_upper

    @torch.inference_mode()
    def local_search(
        self,
        model,
        upper: Tensor,
        lower: Tensor,
        is_upper: Tensor,
        split: int,
        y: Tensor,
        criterion,
        start: int,
        end: int,
        iter: int,
    ):
        config = config_parser.config
        for _ in range(config.iteration):
            is_upper = self.insert(is_upper)
            is_upper = self.deletion(is_upper)

        _is_upper = is_upper.repeat([1, 1, split, split]).to(config.device)
        x_adv = upper * _is_upper + lower * ~_is_upper
        logits = model(x_adv).detach().clone()
        loss1 = criterion(logits, y).detach().clone()
        self._robust_acc(logits, y, loss1, start, end, iter)

        x_adv = upper * ~_is_upper + lower * _is_upper
        logits = model(x_adv).detach().clone()
        loss2 = criterion(logits, y).detach().clone()
        self._robust_acc(logits, y, loss2, start, end, iter)

    def insert(self):
        max_heap = []
        # for e in

    def deletion(self):
        pass

    def split_block(self, split):
        split *= 2
        return split
