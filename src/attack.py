from argparse import ArgumentParser

import utils
from attacker import get_attacker
from base import get_criterion, get_model, load_dataset
from utils import config_parser, rename_dir, reproducibility, setup_logger


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
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="10:DEBUG,20:INFO,30:WARNING,40:ERROR,50:CRITICAL",
    )
    args = parser.parse_args()
    utils.DEBUG = args.debug
    return args


def main():
    reproducibility()
    criterion = get_criterion()
    for model_name, batch_size in config.model.items():
        print(f"\n{model_name}")
        data, label = load_dataset(model_name)
        model = get_model(model_name, batch_size)
        attacker = get_attacker()
        attacker.attack(model, data, label, criterion)


if __name__ == "__main__":
    args = argparser()
    config = config_parser.read(args.config, args)
    config.savedir = rename_dir(f"../result/{config.attacker}")
    config_parser.save(config.savedir + "/config.json")
    logger = setup_logger.setLevel(args.log_level)
    main()

else:
    logger = setup_logger(__name__)
