from argparse import ArgumentParser
from datetime import datetime

import torch

from Attacker import get_attacker
from base import get_model, load_imagenet
from utils import config_parser, reproducibility, setup_logger


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
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
        "--n_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--thread",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=10,
        help="10:DEBUG,20:INFO,30:WARNING,40:ERROR,50:CRITICAL",
    )
    args = parser.parse_args()
    return args


def main():
    torch.set_num_threads(config.thread)
    for model_container, models in config.model.items():
        for model_name, batch_size in models.items():
            print(f"{model_name}")
            reproducibility(config.seed)
            config.datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            attacker = get_attacker()
            model = get_model(model_container, model_name, batch_size)
            data, label = load_imagenet(model_name)
            attacker.attack(model, data, label)


if __name__ == "__main__":
    args = argparser()
    logger = setup_logger.setLevel(args.log_level)
    config = config_parser.read(args.config, args)
    main()

else:
    logger = setup_logger(__name__)
