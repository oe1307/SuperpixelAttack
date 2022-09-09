from robustbench.data import load_cifar10, load_cifar100

from utils import config_parser

from .load_imagenet import load_imagenet


def load_dataset():
    config = config_parser.config

    if config.dataset == "cifar10":
        img, label = load_cifar10(config.n_examples, "../storage/data")
    elif config.dataset == "cifar100":
        img, label = load_cifar100(config.n_examples, "../storage/data")
    elif config.dataset == "imagenet":
        img, label = load_imagenet(config.n_examples, "../data/data")
    else:
        raise ValueError("Dataset not supported")

    return img, label
