import tarfile
from argparse import ArgumentParser

import torch
from robustbench.data import load_imagenet

from attacker import get_attacker
from base import expand_imagenet, get_model
from utils import config_parser, fix_seed, save_error


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
        type=lambda x: int(x) if x.isdigit() else x,
        required=True,
        help="0, 1, ... or 'cpu' or 'mps'",
    )
    parser.add_argument(
        "-t",
        "--thread",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-p",
        "--param",
        type=str,
        nargs="*",
        help="params to override, e.g.) -p iter=100 n_examples=1000",
    )
    args = parser.parse_args()
    return args


@save_error()
def main():
    # setup
    torch.set_num_threads(config.thread)
    fix_seed(config.seed)
    expand_imagenet()

    # backup
    config.set_savedir(f"../result/{config.attacker}/{config.model}/result")
    config_parser.save(f"{config.savedir}/config.json")
    tarfile.open(f"{config.savedir}/backup.tar.gz", "w:gz").add(".")

    # attack
    attacker = get_attacker()
    model, transform = get_model()
    data, label = load_imagenet(config.n_examples, "../storage/data", transform)
    attacker.attack(model, data, label)
    config_parser.save(f"{config.savedir}/config.json")


if __name__ == "__main__":
    config = config_parser.read(argparser())
    main()
