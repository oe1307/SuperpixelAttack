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
            if config.update_method in ("greedy_local_search",):
                assert False
            elif config.update_method in ("uniform_distribution",):
                breakpoint()

        elif config.update_area == "random_square":
            for idx in range(self.batch):
                if forward[idx] >= self.checkpoint[idx]:
                    self.update_area[idx] = np.zeros((self.height, self.width))
                    n_half = (self.half_point < forward[idx]).sum()
                    p = config.p_init / 2**n_half
                    h = np.sqrt(p * self.height * self.width).round().astype(int)
                    r = np.random.randint(0, self.height - h)
                    s = np.random.randint(0, self.width - h)
                    self.update_area[:, r : r + h, s : s + h] = 1

            if config.update_method in ("greedy_local_search",):
                breakpoint()
            elif config.update_method in ("uniform_distribution",):
                self.targets = np.ones(self.batch, dtype=int)[:, None]
                self.checkpoint = forward + 1
            else:
                raise ValueError(config.update_method)

        elif config.update_area == "divisional_square":
            assert False

        else:
            raise NotImplementedError(config.update_area)

        return self.update_area, self.targets
