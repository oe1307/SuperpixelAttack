import datetime
import json
import os
import pprint
import socket
from typing import List

import git
import yaml
from attrdict import AttrDict

from .logging import change_level, setup_logger

logger = setup_logger(__name__)


class ConfigParser:
    def __init__(self):
        self.config = AttrDict()

    def read(self, path: str = None, args=None, param: List[str] = None):
        if path is not None:
            logger.info(f"\n [ READ ] {path}")
            self.config.update(yaml.safe_load(open(path, mode="r")))
        if args is not None:
            args = vars(args)
            assert args.keys() & self.config.keys() == set()
            self.config.update(args)
        if param is not None:
            param = {p.split("=")[0]: eval(p.split("=")[1]) for p in param}
            del self.config["param"]
            assert param.keys() & self.config.keys() == set(param)
            self.config.update(param)
        msg = pprint.pformat(self.config, width=40)
        logger.info(f"{msg}\n")
        return self.config

    def __call__(self):
        return self.config

    def save(self, path: str):
        change_level("git", 30)
        self.config.datetime = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.config.hostname = socket.gethostname()
        git_repo = git.Repo(search_parent_directories=True)
        self.config.branch = git_repo.active_branch.name
        self.config.git_hash = git_repo.head.object.hexsha[:7]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(self.config, open(path, mode="w"), indent=4)


config_parser = ConfigParser()
