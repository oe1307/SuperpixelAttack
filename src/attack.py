from argparse import ArgumentParser
from datetime import datetime

import torch

from Attacker import get_attacker
from Base import get_model, load_dataset
from Utils import config_parser, reproducibility, setup_logger


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
        default=10,
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=10,
        help="10:DEBUG,20:INFO,30:WARNING,40:ERROR,50:CRITICAL",
    )
    parser.add_argument(
        "--exp",
        action="store_true",
    )
    args = parser.parse_args()
    return args


def main():
    torch.set_num_threads(config.thread)
    for model_container, models in config.model.items():
        for model_name, batch_size in models.items():
            print(f"{model_name}")
            reproducibility()
            config.datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            data, label = load_dataset(model_name, data_dir="../storage/data")
            model = get_model(
                model_container, model_name, batch_size, model_dir="../storage/model"
            )
            attacker = get_attacker()
            attacker.attack(model, data, label)


if __name__ == "__main__":
    args = argparser()
    logger = setup_logger.setLevel(args.log_level)
    config = config_parser.read(args.config, args)
    main()

else:
    logger = setup_logger(__name__)
