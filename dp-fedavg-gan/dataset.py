from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from util import HyperParam


class TabularDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self._data = data

    def __getitem__(self, index):
        return torch.tensor(self._data[index], dtype=torch.float32)

    def __len__(self):
        return len(self._data)


def split_dataset(opt: HyperParam, raw_dataset: np.ndarray, split_ratio: Iterable[float]):
    assert np.isclose(sum(split_ratio), 1)
    dataset_size = len(raw_dataset)
    cli_dl = []
    start = 0
    for sr in split_ratio:
        end = start + sr
        ds = TabularDataset(raw_dataset[round(dataset_size * start): round(dataset_size * end)])
        dl = DataLoader(dataset=ds, batch_size=opt.batch_size, shuffle=True, num_workers=2, drop_last=True)
        cli_dl.append(dl)
        start = end
    return cli_dl
