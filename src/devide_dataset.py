import json
import math
import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch import Tensor

from Base import get_criterion, get_model, load_dataset
from Utils import config_parser, pbar, reproducibility, setup_logger


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "--container",
        type=str,
        default="robustbench",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-g",
        "--device",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--step",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03137254901960784,
    )
    args = parser.parse_args()
    return args


def main():
    criterion = get_criterion()
    data, label = load_dataset(config.model, data_dir="../storage/data")
    model = get_model(
        config.container, config.model, config.batch_size, model_dir="../storage/model"
    )
    success_iter = torch.zeros(data.shape[0], device=config.device)

    num_batch = math.ceil(data.shape[0] / model.batch_size)
    for i in range(num_batch):
        start = i * model.batch_size
        end = min((i + 1) * model.batch_size, data.shape[0])
        x = data[start:end].to(config.device)
        y = label[start:end].to(config.device)
        success_iter[start:end] = pgd(model, criterion, x, y)
        torch.cuda.empty_cache()

    savefile = f"../data/{config.dataset}/{config.model}.json"
    success_iter = success_iter.cpu().numpy()
    index = {
        "clean": np.where(success_iter == 0)[0].tolist(),  # 元々誤分類
        "easy": np.where(success_iter == 1)[0].tolist(),  # 1回で成功
        "hard": np.where(np.logical_and(1 < success_iter, success_iter <= config.step))[
            0
        ].tolist(),  # 2回目以降で成功
        "fail": np.where(success_iter == config.step + 1)[0].tolist(),  # 失敗
    }
    os.makedirs(os.path.dirname(savefile), exist_ok=True)
    json.dump(index, open(savefile, "w"), indent=4)


def pgd(model, criterion, x: Tensor, y: Tensor) -> Tensor:
    robust_acc = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    success_iter = torch.zeros(x.shape[0], device=x.device)
    upper = (x + config.epsilon).clamp(0, 1).clone()
    lower = (x - config.epsilon).clamp(0, 1).clone()
    x_adv = x.clone().requires_grad_()
    logits = model(x_adv).clone()
    robust_acc = (logits.argmax(dim=1) == y).clone()
    success_iter[robust_acc] = 1
    loss = criterion(logits, y).sum().clone()
    for step in range(config.step):
        pbar(step + 1, config.step)
        grad = torch.autograd.grad(loss, [x_adv])[0].clone()
        x_adv = (
            (x_adv + 2 * config.epsilon * torch.sign(grad)).clamp(lower, upper).clone()
        )
        del grad
        assert torch.all(x_adv <= upper + 1e-6) and torch.all(x_adv >= lower - 1e-6)
        logits = model(x_adv).clone()
        robust_acc = torch.logical_and(robust_acc, (logits.argmax(dim=1) == y)).clone()
        success_iter += robust_acc
        loss = criterion(logits, y).sum().clone()
    return success_iter


if __name__ == "__main__":
    config = config_parser.read(args=argparser())
    config.criterion = "cw"
    config.norm = "Linf"
    reproducibility()
    main()

else:
    logger = setup_logger(__name__)
