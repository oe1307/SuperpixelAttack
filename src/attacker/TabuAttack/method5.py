import numpy as np
import torch
from torch import Tensor

from base import Transfer
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class TabuAttack5(Transfer):
    """
    - one-flip
    - flipすることを禁止
    - 局所移動戦略
    """

    def __init__(self):
        super().__init__()

    def recorder(self):
        super().recorder()
        self.best_loss = torch.zeros(
            (config.n_examples, config.iteration), device=config.device
        )
        self.current_loss = torch.zeros(
            (config.n_examples, config.iteration), device=config.device
        )

    @torch.no_grad()
    def _attack(self, x_all: Tensor, y_all: Tensor):
        """
        x_best: 過去の探索で最も良かった探索点
        _x_best: 近傍内で最も良かった探索点
        x_adv: 現在の探索点

        is_upper: 現在の探索点
        _is_upper_best: 近傍内で最も良かった探索点
        _is_upper: 前反復の探索点

        best_loss: 過去の探索で最も良かったloss
        loss: 現在のloss
        _best_loss: 近傍内で最も良かったloss
        _loss: 前反復のloss

        flip: 探索するindex
        _flip: tabuに入れるindex

        TODO:
            batch処理
        """
        x_adv_all = []
        for self.idx, (x, y) in enumerate(zip(x_all, y_all)):
            # initialize
            upper = (x + config.epsilon).clamp(0, 1).clone()
            lower = (x - config.epsilon).clamp(0, 1).clone()
            x_best = self.transfer(x.unsqueeze(0), y).squeeze(0).clone()
            _is_upper = (x_best == upper).clone()
            _loss = self.robust_acc(x_best, y).item()
            best_loss = _loss
            self.current_loss[self.idx, 1] = _loss
            self.best_loss[self.idx, 1] = _loss
            tabu_list = -config.tabu_size * torch.ones(x.numel()) - 1

            for iter in range(2, config.iteration):
                _best_loss = -100
                tabu = iter - tabu_list < config.tabu_size
                flips = np.random.choice(np.where(~tabu)[0], config.search)
                for flip in flips:
                    is_upper = _is_upper.clone()
                    is_upper.view(-1)[flip] = ~is_upper.view(-1)[flip]
                    x_adv = torch.where(is_upper, upper, lower).clone()
                    loss = self.robust_acc(x_adv, y).item()
                    if loss > _best_loss:  # 近傍内の最良点
                        _flip = flip
                        _is_upper_best = is_upper.clone()
                        _x_best = x_adv.clone()
                        _best_loss = loss
                    logger.debug(
                        f"( iter={iter} ) loss={loss:.4f} best_loss={_best_loss:.4f}"
                    )
                    if _best_loss > _loss:
                        break

                # end for
                _is_upper = _is_upper_best.clone()
                _loss = _best_loss
                tabu_list[_flip] = iter

                if _best_loss > best_loss:  # 過去の最良点
                    x_best = _x_best.clone()
                    best_loss = _loss

                self.current_loss[self.idx, iter] = _best_loss
                self.best_loss[self.idx, iter] = best_loss

                if not config.exp and _loss > 0:
                    logger.info(f"idx={self.idx} iter={iter} success")
                    break

            assert torch.all(x_adv <= upper + 1e-6) and torch.all(x_adv >= lower - 1e-6)
            x_adv_all.append(x_best)
        x_adv_all = torch.stack(x_adv_all)
        return x_adv_all

    def robust_acc(self, x_adv: Tensor, y: Tensor) -> Tensor:
        """index-wise robust accuracy"""
        assert x_adv.dim() == 3
        x_adv = x_adv.unsqueeze(0)
        logits = self.model(x_adv).clone()
        self.num_forward += 1
        self._robust_acc[self.idx] = torch.logical_and(
            self._robust_acc[self.idx], logits.argmax(dim=1) == y
        )
        self.success_iter[self.idx] += self._robust_acc[self.idx]
        loss = self.criterion(logits, y).clone()
        return loss
