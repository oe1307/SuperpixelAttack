from argparse import ArgumentParser

import yaml

import optuna
import utils
from utils import config_parser, rename_file, reproducibility, setup_logger


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-r",
        "--optuna",
        type=str,
        default="config/optuna.yaml",
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
    breakpoint()
    study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=100)
    savefile = rename_file("../result/optuna/best_params.yaml")
    yaml.dump(study.best_params, open(savefile, "w"))


if __name__ == "__main__":
    args = argparser()
    logger = setup_logger.setLevel(args.log_level)
    config = config_parser.read(args.config, args)
    config = config_parser.read(args.optuna)
    main()
