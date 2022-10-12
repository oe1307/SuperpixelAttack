from argparse import ArgumentParser
from datetime import datetime

import torch

import utils
from attacker import get_attacker
from base import get_criterion, get_model, load_dataset
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
        "--thread",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=10,
        help="10:DEBUG,20:INFO,30:WARNING,40:ERROR,50:CRITICAL",
    )
    args = parser.parse_args()
    utils.DEBUG = args.debug
    return args


def main():
    torch.set_num_threads(config.thread)
    criterion = get_criterion()
    for model_container, models in config.model.items():
        for model_name, batch_size in models.items():
            print(f"{model_name}")
            reproducibility()
            config.savedir = (
                f"../result/{config.attacker}/{config.dataset}/{config.norm}/"
                + f"{model_name}/{datetime.now().strftime('%Y-%m-%d_%H:%M')}/"
            )
            data, label = load_dataset(model_name, data_dir="../storage/data")
            model = get_model(
                model_container, model_name, batch_size, model_dir="../storage/model"
            )
            attacker = get_attacker()
            attacker.attack(model, data, label, criterion)


if __name__ == "__main__":
    args = argparser()
    logger = setup_logger.setLevel(args.log_level)
    config = config_parser.read(args.config, args)
    main()

else:
    logger = setup_logger(__name__)
