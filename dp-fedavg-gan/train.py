from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from DPv1 import DPHandler, dp_handler_factory
from model import Discriminator, Generator
from util import HyperParam, path_join


@dataclass
class NetOptimPair:
    net: Discriminator  # torch.nn.Module
    optim: torch.optim.Optimizer


def create_cli_dis(opt: HyperParam, num_cli: int):
    cli_dis_pair = []
    for _ in range(num_cli):
        d_net = Discriminator(opt).to(opt.device).train()
        d_optim = torch.optim.Adam(d_net.parameters(), lr=opt.dis_lr, betas=(opt.beta1, .999))
        # d_optim = torch.optim.SGD(d_net.parameters(), lr=opt.dis_lr)
        cli_dis_pair.append(NetOptimPair(d_net, d_optim))
    return cli_dis_pair


@dataclass
class CliWeightDelta:
    delta: Dict[str, torch.Tensor]
    batch: int


def cli_dis_upd(opt: HyperParam,
                dis_weight: dict,
                generator: Generator,
                dl: DataLoader,
                discriminator: Discriminator,
                d_optim: torch.optim.Optimizer):
    discriminator.load_state_dict(dis_weight)  # function does no-grad in-place copy
    loss = torch.nn.BCELoss().to(opt.device)
    for batch_idx, data in enumerate(dl):
        d_optim.zero_grad()

        noise = torch.randn(opt.batch_size, opt.latent_size, device=opt.device)
        fake = generator(noise).detach()
        fake_critic = discriminator(fake)
        fake_label = torch.zeros(fake_critic.shape, device=opt.device)
        fake_loss = loss(fake_critic, fake_label)

        real = data.to(opt.device)
        real_critic = discriminator(real)
        real_label = torch.ones(real_critic.shape, device=opt.device)
        real_loss = loss(real_critic, real_label)

        d_loss = fake_loss + real_loss
        d_loss.backward()
        d_optim.step()

        if batch_idx + 1 >= opt.dis_step_max:
            break
    batch_cnt = batch_idx + 1
    print(f'loss D {d_loss:.6f}; E[D(x)] {real_critic.mean():.6f}, E[D(G(z))] {fake_critic.mean():.6f}')

    # a new dict whose tensors are detached references
    weight_delta = discriminator.state_dict()
    with torch.no_grad():
        for name in weight_delta:
            delta = weight_delta[name] - dis_weight[name]  # DO NOT use inplace -=

            # legacy DP L2 clip
            dp_l2_clip = opt.dp_l2_bound
            clip = torch.min(torch.tensor(1., device=opt.device), dp_l2_clip / delta.norm(p=2))
            delta *= clip

            # if opt.use_dp:
            #     clip = torch.min(torch.tensor(1., device=opt.device), opt.dp_l2_clip / delta.norm(p=2))
            #     delta *= clip
            #     # do noising before normalization
            #     noise = torch.normal(mean=0., std=opt.dp_noise_scale * opt.dp_l2_clip,
            #                          size=delta.shape, device=opt.device)
            #     delta += noise

            weight_delta[name] = delta
    return CliWeightDelta(weight_delta, batch_cnt), d_loss.item()


def fed_avg(weights: List[CliWeightDelta], weighted: bool = False):
    w_avg = {name: state.detach().clone() for name, state in weights[0].delta.items()}
    if len(weights) == 1:
        return w_avg

    sum_batch = sum(w.batch for w in weights) if weighted else len(weights)
    with torch.no_grad():
        for name in w_avg:
            if weighted:
                w_avg[name] *= weights[0].batch
            for cli in range(1, len(weights)):
                w = weights[cli]
                if weighted:
                    w_avg[name] += w.delta[name] * w.batch
                else:
                    w_avg[name] += w.delta[name]
            w_avg[name] /= sum_batch
    return w_avg


def dis_upd(opt: HyperParam,
            ser_dis: Discriminator,
            ser_gen: Generator,
            cli_dl: List[DataLoader],
            cli_dis_pair: List[NetOptimPair],
            dp_handler: DPHandler):
    old_weight = ser_dis.state_dict()

    # cli_idx = np.random.choice(len(cli_dl), size=max(1, int(opt.client_frac * len(cli_dl))), replace=False)
    cli_idx = np.random.permutation(len(cli_dl))
    cli_weight_delta = []

    cli_d_loss = []

    for cli in cli_idx:
        print(f'cli#{cli}: ', end='')
        pair = cli_dis_pair[cli]
        weight_delta, d_loss = cli_dis_upd(opt, old_weight, ser_gen, cli_dl[cli], pair.net, pair.optim)
        cli_weight_delta.append(weight_delta)
        cli_d_loss.append(d_loss)

    weight_delta = fed_avg(cli_weight_delta, weighted=opt.fedavg_weighted)

    new_weight = dict()
    with torch.no_grad():
        for name in weight_delta:
            # new_weight[name] = old_weight[name] + weight_delta[name]

            grad = dp_handler.compute_sanitized_gradients(weight_delta[name])
            new_weight[name] = old_weight[name] + grad
    ser_dis.load_state_dict(new_weight)

    return np.mean(cli_d_loss)


def gen_upd(opt: HyperParam,
            discriminator: Discriminator,
            generator: Generator,
            g_optim: torch.optim.Optimizer):
    loss = torch.nn.BCELoss().to(opt.device)
    for _ in range(opt.gen_step):
        g_optim.zero_grad()
        discriminator.zero_grad()
        noise = torch.randn(opt.batch_size, opt.latent_size, device=opt.device)
        fake = generator(noise)
        fake_critic = discriminator(fake)
        real_label = torch.ones(fake_critic.shape, device=opt.device)
        g_loss = loss(fake_critic, real_label)
        g_loss.backward()
        g_optim.step()
    print(f'loss G {g_loss:.6f}; E[D(G(z))] {fake_critic.mean():.6f}')
    return g_loss.item(), fake_critic.mean().item()


def train(opt: HyperParam, cli_dl: List[DataLoader], *, evaluator=None):
    ser_dis = Discriminator(opt).to(opt.device)
    if opt.dis_sdict:
        sdict = torch.load(opt.dis_sdict, map_location=opt.device)
        ser_dis.load_state_dict(sdict)
    ser_dis.train()
    cli_dis_pair = create_cli_dis(opt, len(cli_dl))

    ser_gen = Generator(opt).to(opt.device)
    if opt.gen_sdict:
        sdict = torch.load(opt.gen_sdict, map_location=opt.device)
        ser_gen.load_state_dict(sdict)
    ser_gen.train()
    g_optim = torch.optim.Adam(ser_gen.parameters(), lr=opt.gen_lr, betas=(opt.beta1, .999))
    # g_optim = torch.optim.SGD(ser_gen.parameters(), lr=opt.gen_lr)

    print(ser_dis)
    print(ser_gen)

    # [DP] create handler
    dp_handler = dp_handler_factory(opt)

    if evaluator is not None:
        evaluator.set_gen(ser_gen)

    for epoch in range(opt.n_epoch):
        print(f'#{epoch:3d}')

        d_loss = dis_upd(opt, ser_dis, ser_gen, cli_dl, cli_dis_pair, dp_handler)

        g_loss, _ = gen_upd(opt, ser_dis, ser_gen, g_optim)

        if epoch % opt.save_step == 0 or epoch == opt.n_epoch - 1:
            torch.save(ser_dis.state_dict(), path_join(opt.out_folder, f'ffd_{epoch:03d}.pt'))
            torch.save(ser_gen.state_dict(), path_join(opt.out_folder, f'ffg_{epoch:03d}.pt'))

        if evaluator is not None:
            evaluator.collect_loss(d_loss, g_loss)
            evaluator.analyse_ml_performance(epoch)

        # [DP] check budget
        eps_delta = dp_handler.get_budget(opt.dp_delta)
        print(f'[DP] epsilon: {eps_delta.spent_eps:.6f}; delta: {eps_delta.spent_delta:.6f}')
        if eps_delta.spent_delta > opt.dp_delta or eps_delta.spent_eps > opt.dp_epsilon:
            print(f'[DP] terminate at epoch {epoch:03d}')
            opt.n_epoch = epoch + 1  # useless. hope to avoid some bug
            break

    if evaluator is not None:
        evaluator.plot_loss()
        evaluator.plot_ml_info()

    return ser_gen
