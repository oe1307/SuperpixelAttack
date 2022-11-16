import os
from typing import Union

import torch
from robustbench.data import CustomImageFolder, get_preprocessing
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from torch import Tensor

from utils import config_parser, pbar

config = config_parser()


def load_imagenet(model_name: str, data_dir: str) -> Union[Tensor, Tensor]:
    """Load ImageNet data"""
    data_dir = os.path.join(data_dir, "imagenet/val")
    if not os.path.exists(data_dir):
        raise FileNotFoundError("please download imagenet dataset")
    transform = get_preprocessing(
        BenchmarkDataset.imagenet, ThreatModel(config.norm), model_name, None
    )
    dataset = CustomImageFolder(data_dir, transform=transform)
    img = list()
    label = list()
    for index in range(config.n_examples):
        pbar(index + 1, config.n_examples, "Loading ImageNet data")
        x, y = dataset.__getitem__(index)[:2]
        img.append(x)
        label.append(y)
    img = torch.stack(img)
    label = torch.tensor(label)
    return img, label
