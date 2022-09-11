from argparse import ArgumentParser

import utils
from utils import config_parser, reproducibility, setup_logger


def argparser():
    parser = ArgumentParser()
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


if __name__ == "__main__":
    args = argparser()
    base_config = config_parser.read(args.config, args)
    main()
