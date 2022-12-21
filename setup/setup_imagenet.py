import os
from scipy.io import loadmat

devkit_dir = "../storage/data/ILSVRC2012_devkit_t12/data/"

meta = loadmat(devkit_dir + "meta.mat", squeeze_me=True)
ILSVRC2012_ID_to_WNID = {m[0]: m[1] for m in meta["synsets"]}

f = open(devkit_dir + "ILSVRC2012_validation_ground_truth.txt")
ILSVRC2012_IDs = (int(ILSVRC2012_ID) for ILSVRC2012_ID in f.read().splitlines())
f.close()

num_valid_images = 50000
for valid_id, ILSVRC2012_ID in enumerate(ILSVRC2012_IDs):
    WNID = ILSVRC2012_ID_to_WNID[ILSVRC2012_ID]
    filename = f"ILSVRC2012_val_{valid_id + 1:0>8}.JPEG"
    os.makedirs(f"../storage/data/imagenet/{WNID}", exist_ok=True)
    os.rename(
        f"../storage/data/imagenet/{filename}",
        f"../storage/data/imagenet/{WNID}/{filename}",
    )
