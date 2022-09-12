import torch

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

        for i in range(1, config.iteration):
            logger.info(f"   iteration {i}")
            x_adv = self.local_search()
            self._robust_acc(model, x_adv, y, criterion, start, end, iter=1)
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

    def local_search(self):
        pass

    def insert(self):
        pass

    def deletion(self):
        pass

    def split_block(self, split):
        split *= 2
        return split
