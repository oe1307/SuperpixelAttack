import time
from datetime import datetime

import numpy as np
import torch

from utils import config_parser, setup_logger

logger = setup_logger(__name__)


class Recorder:
    def __init__(self):
        self.recorder()

    def recorder(self):
        """This function is for recording the information of the attacker."""
        config = config_parser.config
        logger.info(f"Attack start at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

        self.total_time = time.time()
        self._clean_acc = 0
        self._robust_acc = torch.zeros(
            config.n_examples,
            dtype=torch.bool,
            device=config.device,
        )
        self.best_loss = torch.zeros(
            (config.n_examples, config.iteration + 1),
            dtype=torch.float16,
            device=config.device,
        )
        self.current_loss = torch.zeros(
            (config.n_examples, config.iteration + 1),
            dtype=torch.float16,
            device=config.device,
        )
        self.num_forward = 0
        self.num_backward = 0
        self._recorder()

    def _recorder(self):
        """This function for override."""
        pass

    def record(self):
        """This function is for recording the information of the attacker."""
        config = config_parser.config
        logger.info(f"Attack start at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

        self.total_time = time.time() - self.total_time
        self._clean_acc = self._clean_acc / config.n_examples * 100
        self._robust_acc = self._robust_acc.sum() / config.n_examples * 100
        self.ASR = 100 - self._robust_acc

        np.save(f"{self.savedir}/best_loss.npy", self.best_loss.cpu().numpy())
        np.save(f"{self.savedir}/current_loss.npy", self.best_loss.cpu().numpy())

        print(
            "\n"
            + f"total time (sec) = {self.total_time:.2f}s\n"
            + f"clean acc (%) = {self._clean_acc:.2f}\n"
            + f"robust acc (%) = {self._robust_acc:.2f}\n"
            + f"ASR (%) = {self.ASR:.2f}\n"
            + f"num_forward = {self.num_forward}\n"
            + f"num_backward = {self.num_backward}\n",
            file=open(self.savedir + "/summary.txt", "w"),
        )
        self._record()

    def _record(self):
        """This function for override."""
        pass
