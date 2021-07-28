import pandas as pd
from torch.utils.data import DataLoader

from dataset import TabularDataset
from transformer import Transformer
from util import HyperParam


def read_biased_split(opt: HyperParam, transformer: Transformer, column_name: str, split_count: int):
    cli_dl = []
    for sidx in range(split_count):
        df = pd.read_csv(f'./{opt.dataset_name}/{column_name}_{sidx}.csv')
        # drop id column
        df.drop([df.columns[0]], axis=1, inplace=True)
        raw_dataset = transformer.transform(df)
        ds = TabularDataset(raw_dataset)
        dl = DataLoader(dataset=ds, batch_size=opt.batch_size, shuffle=True, num_workers=2, drop_last=True)
        cli_dl.append(dl)
    return cli_dl
