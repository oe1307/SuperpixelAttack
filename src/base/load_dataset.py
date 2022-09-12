from robustbench.data import load_cifar10, load_cifar100
from robustbench.model_zoo import model_dicts
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from utils import config_parser

from .load_imagenet import load_imagenet


def load_dataset(model_name):
    """Load dataset for the given model.

    Args:
        model_name (str): model name in robustbench

    Raises:
        ValueError: if the model is not in robustbench

    Returns:
        img: all images
        label: all labels
    """
    config = config_parser.config

    models = model_dicts[BenchmarkDataset(config.dataset)]
    models = models[ThreatModel(config.norm)].keys()
    assert (
        model_name in models
    ), f"{model_name} not in robustbench[{config.dataset}][{config.norm}]"

    if config.dataset == "cifar10":
        img, label = load_cifar10(config.n_examples, "../storage/data")
    elif config.dataset == "cifar100":
        img, label = load_cifar100(config.n_examples, "../storage/data")
    elif config.dataset == "imagenet":
        img, label = load_imagenet(model_name)
    else:
        raise ValueError("Dataset not supported")

    return img, label
