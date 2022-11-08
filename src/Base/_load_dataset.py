import json
import os

import torch
from robustbench.data import CustomImageFolder, get_preprocessing, load_cifar10
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from Utils import config_parser

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
    for index in range(5000):
        x, y = dataset.__getitem__(index)[:2]
        img.append(x)
        label.append(y)
    img = torch.stack(img)
    label = torch.tensor(label)
    config.n_examples = 5000
    return img, label


def load_cifar10_easy(model_name: str, data_dir: str):
    all_img, all_label = load_cifar10(10000, data_dir)
    index = os.path.join("../data/cifar10", f"{model_name}.json")
    index = json.load(open(index))["easy"]
    img = [all_img[i] for i in index]
    label = [all_label[i] for i in index]
    img = torch.stack(img)
    label = torch.tensor(label)
    config.dataset = "cifar10"
    config.target = "easy"
    config.n_examples = len(index)
    return img, label


def load_cifar10_hard(model_name: str, data_dir: str):
    all_img, all_label = load_cifar10(10000, data_dir)
    index = os.path.join("../data/cifar10", f"{model_name}.json")
    index = json.load(open(index))["hard"]
    img = [all_img[i] for i in index]
    label = [all_label[i] for i in index]
    img = torch.stack(img)
    label = torch.tensor(label)
    config.dataset = "cifar10"
    config.target = "hard"
    config.n_examples = len(index)
    return img, label

def load_imagenet_easy(model_name: str, data_dir: str):
    all_img, all_label = load_imagenet(5000, data_dir)
    index = os.path.join("../data/imagenet", f"{model_name}.json")
    index = json.load(open(index))["easy"]
    img = [all_img[i] for i in index]
    label = [all_label[i] for i in index]
    img = torch.stack(img)
    label = torch.tensor(label)
    config.dataset = "imagenet"
    config.target = "easy"
    config.n_examples = len(index)
    return img, label

def load_imagenet_hard(model_name: str, data_dir: str):
    all_img, all_label = load_imagenet(5000, data_dir)
    index = os.path.join("../data/imagenet", f"{model_name}.json")
    index = json.load(open(index))["hard"]
    img = [all_img[i] for i in index]
    label = [all_label[i] for i in index]
    img = torch.stack(img)
    label = torch.tensor(label)
    config.dataset = "imagenet"
    config.target = "hard"
    config.n_examples = len(index)
    return img, label
