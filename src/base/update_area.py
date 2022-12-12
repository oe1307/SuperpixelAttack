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
                for idx in range(self.batch):
                    if forward[idx] >= self.checkpoint[idx]:
                        self.level[idx] = min(
                            self.level[idx] + 1, len(config.segments) - 1
                        )
                        self.update_area[idx] = self.superpixel[idx, self.level[idx]]
                        self.n_update_area = self.update_area[idx].max()
                        chanel = np.tile(np.arange(self.n_chanel), self.n_update_area)
                        labels = np.repeat(
                            range(1, self.n_update_area + 1), self.n_chanel
                        )
                        self.targets[idx] = np.stack([chanel, labels], axis=1)
                        np.random.shuffle(self.targets[idx])
                        self.checkpoint[idx] += self.n_update_area * self.n_chanel
                    else:
                        self.targets[idx] = np.delete(self.targets[idx], 0, axis=0)
            elif config.update_method in ("uniform_distribution",):
                breakpoint()

        elif config.update_area == "random_square":
            self.update_area = np.zeros(
                (self.batch, self.height, self.width), dtype=int
            )
            half_point = (
                np.array([0.001, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 0.8]) * config.steps
            )
            for idx in range(self.batch):
                n_half = (half_point < forward[idx]).sum()
                p = config.p_init / 2**n_half
                h = np.sqrt(p * self.height * self.width).round().astype(int)
                r = np.random.randint(0, self.height - h)
                s = np.random.randint(0, self.width - h)
                self.update_area[:, r : r + h, s : s + h] = 1

            if config.update_method in ("greedy_local_search",):
                breakpoint()
            elif config.update_method in ("uniform_distribution",):
                self.targets = np.ones(self.batch, dtype=int)[:, None]
            else:
                raise ValueError(config.update_method)

        elif config.update_area == "divisional_square":
            assert False

        else:
            raise NotImplementedError(config.update_area)

        return self.update_area, self.targets
