from robustbench.data import load_cifar10, load_cifar100

from Utils import config_parser, setup_logger

from ._load_dataset import load_cifar10_easy, load_cifar10_hard, load_imagenet

logger = setup_logger(__name__)
config = config_parser()


def load_dataset(model_name, data_dir):
    """Load dataset for the given model"""

    if config.dataset == "cifar10":
        config.n_classes = 10
        if config.epsilon != 8 / 255:
            logger.warning(f"dataset={config.dataset} epsilon={config.epsilon}")
        img, label = load_cifar10(10000, data_dir)
    elif config.dataset == "cifar100":
        config.n_classes = 100
        if config.epsilon != 8 / 255:
            logger.warning(f"dataset={config.dataset} epsilon={config.epsilon}")
        img, label = load_cifar100(10000, data_dir)
    elif config.dataset == "imagenet":
        config.n_classes = 1000
        if config.epsilon != 4 / 255:
            logger.warning(f"dataset={config.dataset} epsilon={config.epsilon}")
        img, label = load_imagenet(model_name, data_dir)
    elif config.dataset == "cifar10_easy":
        config.n_classes = 10
        if config.epsilon != 8 / 255:
            logger.warning(f"dataset={config.dataset} epsilon={config.epsilon}")
        img, label = load_cifar10_easy(model_name, data_dir)
    elif config.dataset == "cifar10_hard":
        config.n_classes = 10
        if config.epsilon != 8 / 255:
            logger.warning(f"dataset={config.dataset} epsilon={config.epsilon}")
        img, label = load_cifar10_hard(model_name, data_dir)
    else:
        raise ValueError("Dataset not supported")

    return img, label
