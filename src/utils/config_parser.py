import datetime
import json
import os
import pprint
import subprocess

import yaml

from .color_print import printc


class ConfigParser(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self.savedir = "../result"
        os.makedirs(self.savedir, exist_ok=True)

    def read(self, args):
        self.update(vars(args))
        if "config" in self:
            printc("green", f"\n [ READ ] {self.config}")
            obj = yaml.safe_load(open(self.config, mode="r"))
            assert self.keys() & obj.keys() == set()
            self.update(obj)
        if "param" in self and self["param"] is not None:
            for param in self["param"]:
                key, value = param.split("=")
                assert key in self, f"Invalid parameter: {key}"
                try:
                    self[key] = eval(value)
                except NameError:
                    self[key] = value
                key_type = str(type(self[key])).split("'")[1]
                printc("yellow", f"{key} = {value}: {key_type}")
        printc("green", "\n", self, "\n")
        return self

    def set_savedir(self, path: str):
        dir_name = path.rstrip("/")
        count = 1
        path = f"{dir_name}{count}/"
        while os.path.exists(path):
            count += 1
            path = f"{dir_name}{count}/"
        os.makedirs(path, exist_ok=True)
        printc("cyan", f"[ SAVEDIR ] {path}")
        self.savedir = path

    def __str__(self):
        obj = dict(self).copy()
        if "param" in obj:
            del obj["param"]
        return pprint.pformat(obj, width=40)

    def __call__(self):
        return self

    def save(self, path: str):
        self.datetime = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.hostname = os.uname()[1]
        _git_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.branch = (
            _git_branch.stdout.decode().strip() if _git_branch.stdout else "unknown"
        )
        _git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.hash = _git_hash.stdout.decode().strip() if _git_hash.stdout else "unknown"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(self, open(path, mode="w"), indent=4)


config_parser = ConfigParser()
