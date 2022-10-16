import json
import os

import torch
from robustbench.data import CustomImageFolder, get_preprocessing, load_cifar10
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from utils import config_parser

config = config_parser()


def load_imagenet(model_name: str, data_dir: str):
    """Load ImageNet data"""
    data_dir = os.path.join(data_dir, "imagenet/val")
    assert os.path.exists(data_dir), "download imagenet dataset"
    transform = get_preprocessing(
        BenchmarkDataset.imagenet, ThreatModel(config.norm), model_name, None
    )
    dataset = CustomImageFolder(data_dir, transform=transform)
    img = list()
    label = list()
    for index in range(config.n_examples):
        x, y = dataset.__getitem__(index)[:2]
        img.append(x)
        label.append(y)
    img = torch.stack(img)
    label = torch.tensor(label)
    return img, label


def load_cifar10_easy(n_examples: int, data_dir: str):
    all_img, all_label = load_cifar10(10000, data_dir)
    index = os.path.join(data_dir, "cifar10.json")
    assert os.path.exists(index)
    index = json.load(open(index))["easy"]
    img = [all_img[index[i]] for i in range(n_examples)]
    label = [all_label[index[i]] for i in range(n_examples)]
    img = torch.stack(img)
    label = torch.tensor(label)
    return img, label


def load_cifar10_hard(n_examples: int, data_dir: str):
    all_img, all_label = load_cifar10(10000, data_dir)
    index = os.path.join(data_dir, "cifar10.json")
    assert os.path.exists(index)
    index = json.load(open(index))["hard"]
    img = [all_img[index[i]] for i in range(n_examples)]
    label = [all_label[index[i]] for i in range(n_examples)]
    img = torch.stack(img)
    label = torch.tensor(label)
    return img, label
