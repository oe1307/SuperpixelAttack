import datetime
import json
import os
import pprint
import socket

import git
import yaml
from attrdict import AttrDict

from .logging import change_level, setup_logger

logger = setup_logger(__name__)


class ConfigParser:
    def __init__(self):
        self.config = AttrDict()

    def read(self, path=None, args=None):
        reader = dict()
        if path is not None:
            logger.info(f"\n [ READ ] {path}")
            reader.update(yaml.safe_load(open(path, mode="r")))
        if args is not None:
            args = vars(args)
            for k in args.keys():
                if args[k] is None and k in reader.keys():
                    args[k] = reader[k]
            self.config.update(args)
        msg = pprint.pformat(reader, width=40)
        logger.info(f"{msg}\n")
        self.config.update(reader)
        return self.config

    def __call__(self):
        return self.config

    def save(self, path):
        change_level("git", 30)
        self.config.datetime = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.config.hostname = socket.gethostname()
        self.config.git_hash = git.cmd.Git("./").rev_parse("HEAD")[:7]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(self.config, open(path, mode="w"), indent=4)


config_parser = ConfigParser()
