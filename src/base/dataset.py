import os

from torchvision.datasets.imagenet import parse_devkit_archive, parse_val_archive


def expand_imagenet():
    """Decompress imagenet dataset"""

    if os.path.exists("../storage/data/val"):
        return
    print("Expanding imagenet dataset...")
    if not os.path.exists("../storage/data/ILSVRC2012_devkit_t12.tar.gz"):
        raise FileExistsError("ILSVRC2012_devkit_t12.tar.gz not found")
    parse_devkit_archive("../storage/data")
    if not os.path.exists("../storage/data/ILSVRC2012_img_val.tar"):
        raise FileExistsError("ILSVRC2012_img_val.tar not found")
    parse_val_archive("../storage/data")
