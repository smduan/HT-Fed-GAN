import torch
import torch.nn as nn

from util import HyperParam


class Discriminator(nn.Module):
    def __init__(self, opt: HyperParam):
        super().__init__()
        self.in_dim = opt.feature_dim * opt.pac_size
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, 256),
            nn.LeakyReLU(.2),
            nn.Dropout(.5),
            nn.Linear(256, 256),
            nn.LeakyReLU(.2),
            nn.Dropout(.5),
            # nn.Linear(256, 256),
            # nn.LeakyReLU(.2),
            # nn.Dropout(.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, i):
        # i.shape == (batch_size, feature_dim)
        o = self.fc(i.view(-1, self.in_dim))  # (batch_size / pac_size, 1)
        return o.view(-1)


class Residual(nn.Module):
    def __init__(self, in_feature: int, out_feature: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_feature, out_feature),
            nn.BatchNorm1d(out_feature, track_running_stats=False),
            nn.ReLU()
        )

    def forward(self, i):
        o = self.fc(i)
        return torch.cat([o, i], dim=1)


class Generator(nn.Module):
    def __init__(self, opt: HyperParam):
        super().__init__()
        self.fc = nn.Sequential(
            Residual(opt.latent_size, 256),
            Residual(opt.latent_size + 256, 256),
            Residual(opt.latent_size + 512, 256),
            Residual(opt.latent_size + 768, 256),
            nn.Linear(opt.latent_size + 1024, opt.feature_dim),
            nn.Tanh()
        )

    def forward(self, i):
        # i.shape == (batch_size, latent_size)
        return self.fc(i)
