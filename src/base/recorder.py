import time
from datetime import datetime

import numpy as np
import torch

from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class Recorder:
    def __init__(self):
        self.recorder()

    def recorder(self):
        """This function is for recording the information of the attacker."""
        logger.info(f"Attack start at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

        self.total_time = time.time()
        self._clean_acc = 0
        self._robust_acc = torch.zeros(
            config.n_examples,
            dtype=torch.bool,
            device=config.device,
        )
        self.success_iter = torch.zeros(
            config.n_examples,
            dtype=torch.int16,
            device=config.device,
        )
        self.num_forward = 0
        self.num_backward = 0

    def record(self):
        """This function is for recording the information of the attacker."""
        logger.info(f"Attack end at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
        config_parser.save(config.savedir + "config.json")

        self.total_time = time.time() - self.total_time
        self._clean_acc = self._clean_acc / config.n_examples * 100
        self._robust_acc = self._robust_acc.sum() / config.n_examples * 100
        self.ASR = 100 - self._robust_acc

        np.save(config.savedir + "best_loss.npy", self.best_loss.cpu().numpy())
        np.save(config.savedir + "current_loss.npy", self.best_loss.cpu().numpy())

        msg = (
            "\n"
            + f"total time (sec) = {self.total_time:.2f}s\n"
            + f"clean acc (%) = {self._clean_acc:.2f}\n"
            + f"robust acc (%) = {self._robust_acc:.2f}\n"
            + f"ASR (%) = {self.ASR:.2f}\n"
            + f"total num_forward = {self.num_forward}\n"
            + f"total num_backward = {self.num_backward}"
        )
        print(msg, file=open(config.savedir + "/summary.txt", "w"))
        logger.info(msg)
