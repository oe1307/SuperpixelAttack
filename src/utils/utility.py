import os
import random
import re
import time
from functools import wraps

import numpy as np

from .logging import setup_logger

logger = setup_logger(__name__)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        wrapper.process_time = time.time() - start
        return result

    return wrapper


def counter(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        wrapper.count += 1
        result = func(*args, **kargs)
        return result

    wrapper.count = 0
    return wrapper


def confirmation(message="Are you sure you want to continue? [y/n]: "):
    while True:
        confirm = input(message)
        if confirm in ["y", "Y", "yes"]:
            break
        elif confirm in ["n", "N", "no", "No"]:
            exit()


def read_log(log_file):
    database = list()
    compiler = re.compile(
        "(?P<status>.)( )+(?P<Expl>[0-9]+)( )+"
        + "(?P<Unexpl>[0-9]+) (?P<Obj>.{10})( ){1,4}"
        + "(?P<Depth>[0-9]*)( ){1,2}(?P<IntInf>[0-9]*)( )+"
        + "(?P<Incumbent>[0-9-.]+)[ ]+(?P<BestBd>[0-9.]+)[ ]+"
        + "(?P<Gap>[0-9.]*)[- %]{1,4}(?P<It_Node>[0-9]*)"
        + "([ -])+(?P<Time>[0-9]+)s"
    )
    for line in open(log_file):
        if (match := compiler.match(line)) is not None:
            database.append(match.groupdict())
    return database


def rename_file(file_path):
    filename, extension = os.path.splitext(file_path)
    count = 1
    file_path = f"{filename}{count}{extension}"
    while os.path.exists(file_path):
        count += 1
        file_path = f"{filename}{count}{extension}"
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name, exist_ok=True)
    logger.debug(f"\n [ SAVE_FILE ] {file_path}")
    return file_path


def rename_dir(dir_path):
    dir_name = dir_path.rstrip("/")
    count = 1
    dir_path = f"{dir_name}{count}/"
    while os.path.exists(dir_path):
        count += 1
        dir_path = f"{dir_name}{count}/"
    os.makedirs(dir_path, exist_ok=True)
    logger.debug(f"[ SAVE_DIR ] {dir_path}")
    return dir_path


def reproducibility(seed=0, use_torch=True):
    """Set random seed for reproducibility."""

    logger.debug(f"[ REPRODUCIBILITY ] seed={seed}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = f"{seed}"
    random.seed(seed)
    np.random.seed(seed)
    if use_torch:
        import torch

        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
