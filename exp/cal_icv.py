import os
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import skimage
from robustbench.data import load_imagenet
from skimage.color import rgb2lab

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from base import expand_imagenet, get_model
from sub import Superpixel
from utils import ProgressBar, config_parser, printc


def argparser():
    parser = ArgumentParser()
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
        default=5000,
    )
    parser.add_argument(
        "-s",
        "--segments_ratio",
        type=int,
        default=4,
    )
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--unconnected",
        action="store_false",
        dest="connected",
    )
    parser.add_argument(
        "-t",
        "--thread",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args


def main():
    def intra_cluster_variation(image, storage, targets):
        image = rgb2lab(image).transpose(2, 0, 1)

        metric = [[], [], []]
        for iter in range(config.iter - 1):
            j = targets[iter, 0]
            color = targets[iter, 1]
            area = targets[iter, 2]
            pixel = storage[j] == area
            variance = np.sum((image[color][pixel] - np.mean(image[color][pixel])) ** 2)
            metric[color].append(np.sqrt(variance) / pixel.sum())
        metric = [np.mean(m) for m in metric]
        pbar.step()
        return np.mean(metric)

    superpixel = Superpixel()
    superpixel.construct(data)
    images = (data.numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    pbar = ProgressBar(config.n_examples, "ICV", color="cyan")
    with ThreadPoolExecutor(config.thread) as executor:
        futures = executor.map(
            intra_cluster_variation, images, superpixel.storage, superpixel.targets
        )
    metric = np.array(list(futures))
    pbar.end()

    message = (
        "\n"
        + f"model: {config.model}\n"
        + f"n_examples: {config.n_examples}\n"
        + f"alpha: {config.alpha}\n"
        + f"connected: {config.connected}\n"
        + f"skimage_version: {skimage.__version__}\n"
        + f"average_icv: {metric.mean() * 100:.2f}%\n"
    )
    print(message + "-" * 10, file=open("../result/icv.txt", "a+"))
    printc("yellow", message)


if __name__ == "__main__":
    config = config_parser.read(argparser())
    config.update({"norm": "Linf", "device": "cpu", "seed": 0})
    expand_imagenet()
    model, transform = get_model()
    print("Loading imagenet...")
    data, label = load_imagenet(config.n_examples, "../storage/data", transform)
    main()
