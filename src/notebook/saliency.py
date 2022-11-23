#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

import argparse
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from robustbench.data import CustomImageFolder, get_preprocessing
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

sys.path.append("../")

from base import SODModel


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters to train your model.")
    parser.add_argument(
        "--imgs_folder",
        default="./data/DUTS/DUTS-TE/DUTS-TE-Image",
        help="Path to folder containing images",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="/work/issa/AdEx_BlackBox/storage/model/saliency/saliency_weight.pth",
        help="Path to model",
        type=str,
    )
    parser.add_argument(
        "--use_gpu", default=True, help="Whether to use GPU or not", type=bool
    )
    parser.add_argument(
        "--img_size", default=256, help="Image size to be used", type=int
    )
    parser.add_argument("--bs", default=24, help="Batch Size for testing", type=int)

    return parser.parse_args()


def run_inference(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device="cuda")
    else:
        device = torch.device(device="cpu")

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt["model"])
    model.to(device)
    model.eval()

    # inf_data = InfDataloader(img_folder=args.imgs_folder, target_size=args.img_size)
    # Since the images would be displayed to the user, the batch_size is set to 1
    # Code at later point is also written assuming batch_size = 1, so do not change
    transform = get_preprocessing(
        BenchmarkDataset.imagenet, ThreatModel("Linf"), "Salman2020Do_R18", None
    )
    dataset = CustomImageFolder(
        "/work/issa/AdEx_BlackBox/storage/data/imagenet/val", transform=transform
    )

    print("Press 'q' to quit.")
    with torch.no_grad():
        for batch_idx, (img_tor, _, _) in enumerate(dataset):
            # if batch_idx not in (
            #     1,
            #     22,
            #     40,
            #     84,
            #     96,
            #     105,
            # ):
            #     continue
            img_tor = img_tor.to(device)
            pred_masks, _ = model(img_tor.unsqueeze(0))

            # Assuming batch_size = 1
            img_tor = img_tor.cpu().numpy() * 255
            img_tor = img_tor.transpose(1, 2, 0).astype(np.uint8)
            pred_masks_raw = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
            pred_masks_round = np.squeeze(pred_masks.round().cpu().numpy(), axis=(0, 1))

            print("Image :", batch_idx)

            plt.subplots(figsize=(8, 8))
            plt.tick_params(
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            plt.imshow(img_tor)
            plt.savefig(f"img{batch_idx}.png")
            plt.close()
            plt.subplots(figsize=(8, 8))
            plt.tick_params(
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            plt.imshow(pred_masks_raw, cmap="gray")
            plt.savefig(f"pred_masks_raw{batch_idx}.png")
            plt.close()
            plt.subplots(figsize=(8, 8))
            plt.tick_params(
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            plt.imshow(pred_masks_round, cmap="gray")
            plt.savefig(f"pred_masks_round{batch_idx}.png")
            plt.close()

            if batch_idx == 10:
                quit()


if __name__ == "__main__":
    rt_args = parse_arguments()
    run_inference(rt_args)


# In[ ]:
