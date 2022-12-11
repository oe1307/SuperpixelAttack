import math

import numpy as np
import torch
from torch import Tensor

from base import Attacker, UpdateArea, UpdateMethod, get_criterion
from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class ProposedMethod(Attacker):
    def __init__(self):
        config.n_forward = config.steps
        self.criterion = get_criterion()
        self.update_area = UpdateArea()
        self.update_method = UpdateMethod()

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        x_adv_all = []
        n_images, n_chanel = x_all.shape[:2]
        n_batch = math.ceil(n_images / self.model.batch_size)
        for b in range(n_batch):
            start = b * self.model.batch_size
            end = min((b + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            y = y_all[start:end]
            batch = np.arange(x.shape[0])
            upper = (x + config.epsilon).clamp(0, 1).clone()
            lower = (x - config.epsilon).clamp(0, 1).clone()

            update_area = self.update_area.initialize(x)
            is_upper, x_adv, loss, forward = self.update_method.initialize(
                x, y, lower, upper
            )

            # local search
            searched = [[] for _ in batch]
            loss_storage = []
            best_loss_storage = [best_loss.cpu().numpy()]
            tabu_list = [np.zeros(n) for n in n_targets]
            for forward in range(1, config.steps + 1):
                is_upper = is_upper_best.clone()
                for idx in batch:
                    if forward >= checkpoint[idx]:
                        # decide attention pixel
                        _loss = np.array(loss_storage)
                        _loss = _loss[pre_checkpoint[idx] - 1 :, idx]
                        _best_loss = np.array(best_loss_storage)
                        _best_loss = _best_loss[pre_checkpoint[idx] - 1 : -1, idx]
                        diff = _loss - _best_loss
                        if config.attention_pixel == "best":
                            target_order = np.argsort(diff)[::-1]
                        elif config.attention_pixel == "worst":
                            target_order = np.argsort(diff)
                        elif config.attention_pixel == "impacter":
                            target_order = np.argsort(np.abs(diff))[::-1]
                        elif config.attention_pixel == "non_impacter":
                            target_order = np.argsort(np.abs(diff))
                        elif config.attention_pixel == "random":
                            target_order = np.arange(len(diff))
                            np.random.shuffle(target_order)
                        assert target_order.shape[0] == np.array(searched[idx]).shape[0]
                        ratio = int(target_order.shape[0] * config.ratio)
                        attention_pixel = np.array(searched[idx])[target_order[:ratio]]
                        # TODO: doubled searched

                        # extract target pixel
                        level[idx] = min(level[idx] + 1, len(config.segments) - 1)
                        pre_superpixel = superpixel[idx].copy()
                        superpixel[idx] = superpixel_storage[idx, level[idx]]
                        label_pair = np.stack(
                            [pre_superpixel.reshape(-1), superpixel[idx].reshape(-1)]
                        ).T
                        pair, count = np.unique(label_pair, axis=0, return_counts=True)
                        targets[idx] = []
                        for c, label in attention_pixel:
                            _pair = pair[pair[:, 0] == label]
                            _count = count[pair[:, 0] == label]
                            _target = _pair[_count >= np.average(_count)]
                            _target[:, 0] = c
                            targets[idx].append(_target)
                        targets[idx] = np.concatenate(targets[idx])
                        n_targets[idx] = targets[idx].shape[0]
                        tabu_list[idx] = np.zeros(n_targets[idx])
                        pre_checkpoint[idx] = checkpoint[idx]
                        checkpoint[idx] += int(n_targets[idx] * config.checkpoint)
                        searched[idx] = []
                    order = np.arange(n_targets[idx])
                    np.random.shuffle(order)
                    for t in order:
                        c, label = targets[idx][t]
                        if tabu_list[idx][t] > 0:
                            tabu_list[idx][t] -= 1
                        else:
                            searched[idx].append((c, label))
                            is_upper[idx, c, superpixel[idx] == label] = ~is_upper[
                                idx, c, superpixel[idx] == label
                            ]
                x_adv = torch.where(is_upper, upper, lower)
                pred = self.model(x_adv).softmax(dim=1)
                loss = self.criterion(pred, y)
                loss_storage.append(loss.cpu().numpy())
                update = loss >= best_loss
                x_best[update] = x_adv[update]
                best_loss[update] = loss[update]
                best_loss_storage.append(best_loss.cpu().numpy())
                is_upper_best[update] = is_upper[update]
                pbar.debug(forward, config.steps, "forward", f"batch: {b}")

            x_adv_all.append(x_best)
        x_adv_all = torch.concat(x_adv_all)
        return x_adv_all
