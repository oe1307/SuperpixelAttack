import os

from .logging import setup_logger

logger = setup_logger(__name__)


def rename_file(file_path):
    filename, extension = os.path.splitext(file_path)
    count = 1
    file_path = f"{filename}{count}{extension}"
    while os.path.exists(file_path):
        count += 1
        file_path = f"{filename}{count}{extension}"
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name, exist_ok=True)
    logger.debug(f"\nsave_file: {file_path}")
    return file_path


def rename_dir(dir_path):
    dir_name = dir_path.rstrip("/")
    count = 1
    dir_path = f"{dir_name}{count}/"
    while os.path.exists(dir_path):
        count += 1
        dir_path = f"{dir_name}{count}/"
    os.makedirs(dir_path, exist_ok=True)
    logger.debug(f"\nsave_dir: {dir_path}")
    return dir_path
