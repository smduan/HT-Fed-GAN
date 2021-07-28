import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from constants import DATASET_DCOL, DATASET_NAME

path_join = os.path.join


@dataclass
class HyperParam:
    dataset_name: str = DATASET_NAME
    train_path: str = './{0}/{0}_train.csv'.format(DATASET_NAME)
    dataset_dcol = DATASET_DCOL[DATASET_NAME]
    syn_path: str = './{}_fake.csv'.format(DATASET_NAME)
    out_folder: str = './saves/'

    dis_sdict: str = ''  # './saves/ffd_{:03d}.pt'.format(40)
    gen_sdict: str = ''  # './saves/ffg_{:03d}.pt'.format(40)

    n_epoch: int = 1000
    save_step: int = 50
    eval_start: int = 100
    eval_step: int = 100

    pac_size: int = 8
    latent_size: int = 128
    batch_size: int = 128

    # lambda_gp: float = 10.
    beta1: float = .5
    # adult use (5e-3, 5e-4)
    # clinical use (2e-3, 2e-3)
    # covtype use (1e-3, 5e-4)
    dis_lr: float = 1e-3
    gen_lr: float = 5e-4

    fedavg_weighted: bool = False

    # use_dp: bool = True
    # dp_l2_clip: float = .1
    # dp_noise_scale: float = .01
    dp_l2_bound: float = 1.
    dp_sigma: float = 1.
    dp_epsilon: float = 9.8
    dp_delta: float = 1e-5

    dis_step_max: int = 10
    gen_step: int = 6

    device: str = 'cpu'

    # auto set by transformer
    dataset_size: int = 0
    feature_dim: int = 0

    def start(self):
        os.makedirs(self.out_folder, exist_ok=True)
        if torch.cuda.is_available():
            cudnn.benchmark = True
            self.device = 'cuda:0'

    @staticmethod
    def set_seed(seed: int = random.randint(1, 0xffff)):
        print(f'seed: {seed}')
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            # cudnn.benchmark = False
            # cudnn.deterministic = True
