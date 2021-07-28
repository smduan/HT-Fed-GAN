from typing import Optional

import numpy as np
import pandas as pd
import torch

from model import Generator
from transformer import Transformer, fit_transform
from util import HyperParam


def synthesize(opt: HyperParam, syn_size: int, gen_weight: dict, transformer: Transformer, save_path: Optional[str]):
    generator = Generator(opt).to(opt.device)
    generator.load_state_dict(gen_weight)
    generator.eval()

    data = np.empty((0, opt.feature_dim))
    with torch.no_grad():
        for _ in range(syn_size // opt.batch_size + 1):
            noise = torch.randn(opt.batch_size, opt.latent_size, device=opt.device)
            fake = generator(noise)
            # fake = fake.view(opt.batch_size, -1)
            # data = np.concatenate([data, fake.cpu().numpy()[:, :opt.feature_dim]], axis=0)
            data = np.concatenate([data, fake.cpu().numpy()], axis=0)
    data = data[:syn_size]

    df = transformer.inverse_transform(data)
    if save_path:
        df.to_csv(save_path, index=False)
        print('finish synthesizing')
    return df


def syn_fake(sdict_path: str):
    opt = HyperParam()
    opt.start()
    transformer, _ = fit_transform(opt, opt.train_path, opt.dataset_dcol)
    sdict = torch.load(sdict_path, map_location=opt.device)
    synthesize(opt, opt.dataset_size, sdict, transformer, opt.syn_path)


def stat_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in df.columns:
        print(f'--- {c} ---')
        print(df[c].value_counts())
        print()
    return df
