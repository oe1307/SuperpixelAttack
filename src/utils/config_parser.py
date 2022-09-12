import datetime
import json
import os
import pprint
import socket

import yaml
from attrdict import AttrDict

from .logging import setup_logger

logger = setup_logger(__name__)


class ConfigParser:
    def __init__(self):
        self.config = AttrDict()
        self.log_format = {
            "indent": 1,
            "width": 40,
            "depth": None,
            "compact": False,
            "sort_dicts": False,
        }

    def read(self, path, args=None):
        logger.debug(f"\n [ READ ] {path}\n")
        obj = yaml.safe_load(open(path, mode="r"))
        if args is not None:
            args = vars(args)
            for k in args.keys():
                if args[k] is None and k in obj.keys():
                    args[k] = obj[k]
            obj.update(args)
        msg = pprint.pformat(obj, **self.log_format)
        logger.debug(msg)
        self.config.update(obj)
        return self.config

    def save(self, path):
        self.config.datetime = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.config.hostname = socket.gethostname()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(self.config, open(path, mode="w"), indent=4)


config_parser = ConfigParser()
