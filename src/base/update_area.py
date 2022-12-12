import numpy as np

from utils import config_parser, setup_logger

from .initial_area import InitialArea

logger = setup_logger(__name__)
config = config_parser()


class UpdateArea(InitialArea):
    def __init__(self):
        super().__init__()

    def next(self, forward: np.ndarray):
        if config.update_area == "superpixel":
            for idx in range(self.batch):
                if forward[idx] >= self.checkpoint[idx]:
                    self.level[idx] = min(self.level[idx] + 1, len(config.segments) - 1)
                    self.update_area[idx] = self.superpixel[idx, self.level[idx]]
                    _n_update_area = self.update_area[idx].max(axis=(1, 2))
                    if config.update_method in ("greedy_local_search",):
                        assert False
                    elif config.update_method in ("uniform_distribution",):
                        self.targets[idx] = np.random.permutation(
                            np.arange(1, _n_update_area + 1)
                        )
                        self.checkpoint[idx] += _n_update_area

        elif config.update_area == "random_square":
            for idx in range(self.batch):
                if forward[idx] >= self.checkpoint[idx]:
                    self.update_area[idx] = np.zeros((self.height, self.width))
                    n_half = (self.half_point < forward[idx]).sum()
                    p = config.p_init / 2**n_half
                    h = np.sqrt(p * self.height * self.width).round().astype(int)
                    r = np.random.randint(0, self.height - h)
                    s = np.random.randint(0, self.width - h)
                    self.update_area[idx, r : r + h, s : s + h] = 1

                    if config.update_method in ("greedy_local_search",):
                        assert False
                    elif config.update_method in ("uniform_distribution",):
                        self.targets[idx] = np.ones(1, dtype=int)[:, None]
                        self.checkpoint = forward + 1
                    else:
                        raise ValueError(config.update_method)

        elif config.update_area == "divisional_square":
            assert False

        else:
            raise NotImplementedError(config.update_area)

        return self.update_area, self.targets
