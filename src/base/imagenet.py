import os
from typing import Callable, Union

from robustbench.data import CustomImageFolder
from torch import Tensor
from torch.utils.data import DataLoader

from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


def load_imagenet(
    transform: Callable, data_dir: str = "../storage/data"
) -> Union[Tensor, Tensor]:
    """Load ImageNet data"""

    config.n_classes = 1000
    data_dir = os.path.join(data_dir, "imagenet")
    if not os.path.exists(data_dir):
        raise FileNotFoundError("please download imagenet dataset")
    dataset = CustomImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, config.n_examples, shuffle=False, num_workers=config.thread
    )
    logger.debug("Loading ImageNet data...")
    data, label = next(iter(dataloader))[:2]
    logger.debug("Loaded ImageNet data")
    data, label = data.to(config.device), label.to(config.device)
    return data, label
