import os
import sys
from argparse import ArgumentParser

from matplotlib import pyplot as plt
from robustbench.data import load_imagenet

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from base import expand_imagenet, get_model
from utils import ProgressBar, config_parser


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--thread",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="Salman2020Do_R18",
    )
    parser.add_argument(
        "-n",
        "--n_examples",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args


def main():
    os.makedirs(f"{config.savedir}/imagenet", exist_ok=True)
    pbar = ProgressBar(config.n_examples, "Save images", color="cyan")
    for idx in range(config.n_examples):
        image = data[idx].numpy().transpose(1, 2, 0)
        plt.subplots(figsize=(8, 8))
        plt.imshow(image)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(f"{config.savedir}/imagenet/idx{idx}.png")
        plt.close()
        pbar.step()
    pbar.end()


if __name__ == "__main__":
    config = config_parser.read(argparser())
    config.update({"device": "cpu", "norm": "Linf"})
    expand_imagenet()
    model, transform = get_model()
    print("Loading imagenet...")
    data, label = load_imagenet(config.n_examples, "../storage/data", transform)
    main()
