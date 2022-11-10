import bisect
import math

import numpy as np
import torch
from torch import Tensor

from Base import Attacker, get_criterion
from Utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class TabuAttack(Attacker):
    def __init__(self):
        super().__init__()
        if config.exp:
            logger.warning("exp mode")
        self.criterion = get_criterion()
        self.n_forward = config.forward
        assert config.strategy in ("fast-fit", "best-fit")
        if config.strategy == "fast-fit":
            logger.warning("strategy = fast-fit")

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        """
        x_best: 過去の探索で最も良かった探索点
        _x_best: 近傍内で最も良かった探索点
        x_adv: 現在の探索点

        _is_upper_best: 近傍内で最も良かった探索点
        is_upper: 現在の探索点
        _is_upper: 前反復の探索点

        best_loss: 過去の探索で最も良かったloss
        _best_loss: 近傍内で最も良かったloss
        loss: 現在のloss
        _loss: 前反復のloss

        flip: 探索するindex
        _flip: tabuに入れるindex
        """
        x_adv_all = []
        for x, y in zip(x_all, y_all):

            # initialize
            upper = (x + config.epsilon).clamp(0, 1)
            lower = (x - config.epsilon).clamp(0, 1)
            _is_upper = torch.randint_like(x, 0, 2, dtype=torch.bool)
            x_best = torch.where(_is_upper, upper, lower)
            _loss = self.criterion(self.model(x_best.unsqueeze(0)), y).item()
            best_loss = _loss
            tabu_list = -config.tabu_size * torch.ones(x.numel()) - 2
            self.forward = 0
            iteration = 0

            while True:
                if self.forward >= config.forward:
                    break
                elif config.exp and best_loss > 1e-6:
                    break
                _best_loss = -100
                iteration += 1
                tabu = iteration - tabu_list < config.tabu_size
                if tabu.sum() > x.numel() / 3:
                    logger.warning("clear tabu list")
                    tabu_list = -config.tabu_size * torch.ones(x.numel()) - 2
                    tabu = torch.zeros_like(tabu, dtype=torch.bool)

                for search in range(config.neighbor_search):
                    if self.forward >= config.forward:
                        break
                    percentage_of_elements = self._get_percentage_of_elements()
                    height_tile = max(
                        int(round(math.sqrt(percentage_of_elements * x.numel() / 3))),
                        1,
                    )
                    N_flip = (height_tile**2) * 3
                    flip = np.random.choice(np.where(~tabu)[0], N_flip, replace=False)
                    is_upper = _is_upper.clone()
                    is_upper.view(-1)[flip] = ~is_upper.view(-1)[flip]
                    x_adv = torch.where(is_upper, upper, lower)
                    loss = self.criterion(self.model(x_adv.unsqueeze(0)), y).item()
                    self.forward += 1
                    assert loss > -100
                    if loss > _best_loss:  # 近傍内の最良点
                        _flip = flip
                        _is_upper_best = is_upper.clone()
                        _x_best = x_adv.clone()
                        _best_loss = loss
                    if config.strategy == "fast-fit" and loss > best_loss:
                        break
                    logger.debug(f"( {iteration=} ) {loss=:.4f} {best_loss=:.4f}")
                # end neighbor search

                _is_upper = _is_upper_best.clone()
                _loss = _best_loss
                tabu_list[_flip] = iteration
                if _best_loss > best_loss:  # 過去の最良点
                    x_best = _x_best.clone()
                    best_loss = _loss

            x_adv_all.append(x_best)
        x_adv_all = torch.stack(x_adv_all)
        # save_file = (
        #     f"../result/tabu_{config.forward}_{config.dataset}"
        #     + f"_{config.target}_{config.epsilon}.npy"
        # )
        # np.save(save_file, x_adv_all.clone().cpu().numpy())
        # quit()
        return x_adv_all

    def _get_percentage_of_elements(self) -> float:  # TODO: hard code
        i_p = self.forward / config.forward
        intervals = [0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
        p_ratio = [0.5**i for i in range(len(intervals) + 1)]
        i_ratio = bisect.bisect_left(intervals, i_p)
        return config.p_init * p_ratio[i_ratio]
