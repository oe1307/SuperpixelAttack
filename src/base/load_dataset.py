from robustbench.data import load_cifar10, load_cifar100

from utils import config_parser

from ._load_dataset import load_cifar10_easy, load_cifar10_hard, load_imagenet

config = config_parser()


def load_dataset(model_name, data_dir):
    """Load dataset for the given model"""

    if config.dataset == "cifar10":
        config.num_classes = 10
        img, label = load_cifar10(10000, data_dir)
    elif config.dataset == "cifar100":
        config.num_classes = 100
        img, label = load_cifar100(10000, data_dir)
    elif config.dataset == "imagenet":
        config.num_classes = 1000
        img, label = load_imagenet(model_name, data_dir)
    elif config.dataset == "cifar10_easy":
        config.num_classes = 10
        img, label = load_cifar10_easy(model_name, data_dir)
    elif config.dataset == "cifar10_hard":
        config.num_classes = 10
        img, label = load_cifar10_hard(model_name, data_dir)
    else:
        raise ValueError("Dataset not supported")

    return img, label
