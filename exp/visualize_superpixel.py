import os
import sys
from argparse import ArgumentParser

from matplotlib import pyplot as plt
from robustbench.data import load_imagenet
from skimage.segmentation import mark_boundaries

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from base import expand_imagenet, get_model
from sub import Superpixel
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
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-s",
        "--segments_ratio",
        type=int,
        default=4,
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--unconnected",
        action="store_false",
        dest="connected",
    )
    args = parser.parse_args()
    return args


def main():
    superpixel = Superpixel()
    superpixel.construct(data)
    n_storage = superpixel.storage.shape[1]

    os.makedirs(f"{config.savedir}/superpixel", exist_ok=True)
    pbar = ProgressBar(config.n_examples, "Save images", color="cyan")
    for idx in range(config.n_examples):
        image = data[idx].permute(1, 2, 0).numpy()
        for j in range(n_storage):
            _superpixel = superpixel.storage[idx, j]
            boundary = mark_boundaries(image, _superpixel)
            plt.subplots(figsize=(8, 8))
            plt.axis("off")
            plt.imshow(boundary)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(f"{config.savedir}/superpixel/idx{idx}_boundary{j}.png")
            plt.close()

            plt.subplots(figsize=(8, 8))
            plt.axis("off")
            plt.imshow(_superpixel)
            plt.title(f"{idx = }, alpha={config.alpha} r={config.segments_ratio}")
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(f"{config.savedir}/superpixel/idx{idx}_{j}.png")
            plt.close()
        pbar.step()
    pbar.end()


if __name__ == "__main__":
    config = config_parser.read(argparser())
    config.update({"device": "cpu", "seed": 0, "norm": "Linf"})
    expand_imagenet()
    model, transform = get_model()
    print("Loading imagenet...")
    data, label = load_imagenet(config.n_examples, "../storage/data", transform)
    main()
