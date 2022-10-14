import numpy as np
import torch
from torch import Tensor

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class TabuAttack2(Attacker):
    """
    - one-flip
    - flipすることを禁止
    - 最良移動戦略
    - 限定選択戦略
    - 長期メモリ(未探索を優先)
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

    @torch.inference_mode()
    def _attack(self, x_all: Tensor, y_all: Tensor):
        """
        x_best: 過去の探索で最も良かった探索点
        _x_best: 近傍内で最も良かった探索点
        x_adv: 現在の探索点

        is_upper_best: 過去の探索で最も良かった探索点
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
            _is_upper = torch.randint_like(x, 0, 2, dtype=torch.bool)
            x_best = torch.where(_is_upper, upper, lower)
            _loss = self.robust_acc(x_best, y).item()
            best_loss = _loss
            self.current_loss[self.idx, 1] = _loss
            self.best_loss[self.idx, 1] = _loss
            tabu_list = -config.tabu_size * torch.ones(x.numel()) - 1
            memory = torch.ones(x.numel()) * config.memory

            for iter in range(2, config.iteration):
                alpha = 0
                _best_loss = -100
                tabu = iter - tabu_list < config.tabu_size
                for i in range(int(config.alpha_max * x.numel())):
                    flip = np.random.choice(
                        np.where(~tabu)[0],
                        p=(memory[~tabu] / (memory[~tabu].sum())).cpu().numpy(),
                    )
                    is_upper = _is_upper.clone()
                    is_upper.view(-1)[flip] = ~is_upper.view(-1)[flip]
                    x_adv = torch.where(is_upper, upper, lower).clone()
                    loss = self.robust_acc(x_adv, y).item()
                    if loss > _best_loss:  # 近傍内の最良点
                        _flip = flip
                        _is_upper_best = is_upper.clone()
                        _x_best = x_adv.clone()
                        _best_loss = loss
                    if _best_loss - _loss > config.beta:
                        alpha += 1
                    else:
                        memory[flip] = max(1, memory[flip] - config.penalty)
                    logger.debug(
                        f"( iter={iter} ) loss={loss:.4f} "
                        + f"best_loss={_best_loss:.4f} alpha={alpha}"
                    )
                    if (
                        alpha > config.alpha_plus * x.numel()
                        and i > config.alpha_min * x.numel()
                    ):
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
