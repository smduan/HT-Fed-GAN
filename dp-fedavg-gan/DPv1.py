import collections
import math
from typing import List

import numpy as np
import torch
import torch.optim as optim

from util import HyperParam

ClipOption = collections.namedtuple('ClipOption', ['l2norm_bound', 'clip'])
EpsDelta = collections.namedtuple('EpsDelta', ['spent_eps', 'spent_delta'])

device = 'cpu'


class GaussianMomentsAccountant:
    def __init__(self, total_examples: int, moment_orders: int = 32):
        self.total_examples = total_examples
        self.moment_orders = range(1, moment_orders + 1)
        self.max_moment_order = max(self.moment_orders)
        self.log_moments = torch.zeros(self.max_moment_order, dtype=torch.float64)
        self.binomial_table = self._generate_binomial_table(self.max_moment_order)

    def accumulate_privacy_spending(self, sigma: float, num_examples: int):
        q = num_examples * 1.0 / self.total_examples
        for i in range(self.max_moment_order):
            moment = self._compute_log_moment(sigma, q, self.moment_orders[i])
            self.log_moments[i].add_(moment)

    def _compute_log_moment(self, sigma: float, q: float, moment_order: int):
        binomial_table = self.binomial_table[moment_order:moment_order + 1, :moment_order + 1]
        qs = torch.exp(
            torch.tensor([i * 1.0 for i in range(moment_order + 1)], dtype=torch.float64)
            * torch.log(torch.tensor(q, dtype=torch.float64))
        )
        moments0 = self._differential_moments(sigma, 0.0, moment_order)
        term0 = torch.sum(binomial_table * qs * moments0)
        moments1 = self._differential_moments(sigma, 1.0, moment_order)
        term1 = torch.sum(binomial_table * qs * moments1)
        return torch.log(q * term0 + (1.0 - q) * term1)

    def _differential_moments(self, sigma: float, s: float, t: int):
        binomial = self.binomial_table[:t + 1, :t + 1]
        signs = np.zeros((t + 1, t + 1), dtype=np.float64)
        for i in range(t + 1):
            for j in range(t + 1):
                signs[i, j] = 1.0 - 2 * ((i - j) % 2)
        exponents = torch.tensor([i * (i + 1.0 - 2.0 * s) / (2.0 * sigma * sigma)
                                  for i in range(t + 1)], dtype=torch.float64)
        x = torch.mul(binomial, torch.from_numpy(signs))
        y = torch.mul(x, torch.exp(exponents))
        z = torch.sum(y, 1)
        return z

    def get_privacy_spent(self, target_deltas: List[float]):
        eps_deltas = []
        for delta in target_deltas:
            log_moments_with_order = zip(self.moment_orders, self.log_moments)
            eps_deltas.append(EpsDelta(self._compute_eps(log_moments_with_order, delta), delta))
        return eps_deltas

    @staticmethod
    def _compute_eps(log_moments, delta):
        min_eps = float('inf')
        for moment_order, log_moment in log_moments:
            if math.isinf(log_moment) or math.isnan(log_moment):
                continue
            min_eps = min(min_eps, (log_moment - math.log(delta)) / moment_order)
        return min_eps

    @staticmethod
    def _generate_binomial_table(m: int):
        table = np.zeros((m + 1, m + 1), dtype=np.float64)
        for i in range(m + 1):
            table[i, 0] = 1
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                v = table[i - 1, j] + table[i - 1, j - 1]
                assert not math.isnan(v) and not math.isinf(v)
                table[i, j] = v
        return torch.from_numpy(table)


class AmortizedGaussianSanitizer:
    def __init__(self, accountant: GaussianMomentsAccountant, default_option: ClipOption):
        self.accountant = accountant
        self.default_option = default_option

    def sanitize(self, x: torch.Tensor, sigma: float,
                 option=ClipOption(None, None), num_examples=None, add_noise=True):
        l2norm_bound, clip = option
        if l2norm_bound is None:
            l2norm_bound, clip = self.default_option
        l2norm_bound_ = torch.tensor(l2norm_bound).to(device)
        if clip:
            x = self._batch_clip_by_l2norm(x, l2norm_bound_)
        if add_noise:
            self.accountant.accumulate_privacy_spending(sigma, num_examples)
            saned_x = self._add_gaussian_noise(x, sigma * l2norm_bound)
        else:
            saned_x = x
        return saned_x

    @staticmethod
    def _batch_clip_by_l2norm(t: torch.Tensor, upper_bound: float):
        batch_size = t.size()[0]
        t2 = torch.reshape(t, (batch_size, -1))
        tensor = torch.tensor([])
        upper_bound_inv = tensor.new_full((batch_size,), 1.0 / upper_bound).to(device)
        l2norm_inv = torch.rsqrt(torch.sum(t2 * t2, 1) + 0.000001).to(device)
        scale = torch.min(upper_bound_inv, l2norm_inv) * upper_bound
        clipped_t = torch.mm(torch.diag(scale), t2)
        clipped_t = torch.reshape(clipped_t, t.size())
        return clipped_t

    @staticmethod
    def _add_gaussian_noise(t: torch.Tensor, sigma: float):
        noisy_t = t + torch.normal(mean=torch.zeros(t.size()), std=sigma).to(device)
        return noisy_t


class DPHandler:
    def __init__(self, sanitizer: AmortizedGaussianSanitizer, sigma: float, batches_per_lot: int = 1):
        self.grad_accum_dict = {}
        self.batches_per_lot = batches_per_lot
        self.sanitizer = sanitizer
        self.sigma = sigma

    @torch.no_grad()
    def compute_sanitized_gradients(self, loss: torch.Tensor):
        px_grads = loss  # TODO: per_example_gradients.
        # now assumes batch_size = 1
        add_noise = True
        sanitized_grads = self.sanitizer.sanitize(
            px_grads, self.sigma,
            num_examples=self.batches_per_lot * px_grads.size()[0],
            add_noise=add_noise
        )
        return sanitized_grads

    @torch.no_grad()
    def get_budget(self, target_delta: float):
        return self.sanitizer.accountant.get_privacy_spent(target_deltas=[target_delta])[0]


def dp_handler_factory(opt: HyperParam):
    global device
    device = opt.device
    accountant = GaussianMomentsAccountant(total_examples=opt.dataset_size)
    sanitizer = AmortizedGaussianSanitizer(accountant, (opt.dp_l2_bound / opt.batch_size, True))
    dp_handler = DPHandler(sanitizer, opt.dp_sigma)
    return dp_handler


class DPAdam(optim.Adam):
    def __init__(self, params, lr, betas=(0.5, 0.9), weight_decay=0, *,
                 sanitizer: AmortizedGaussianSanitizer, sigma: float, batches_per_lot: int = 1):
        super().__init__(params=params, lr=lr, betas=betas, weight_decay=weight_decay)
        self.grad_accum_dict = {}
        self.batches_per_lot = batches_per_lot  # assume 1
        self.sanitizer = sanitizer
        self.sigma = sigma
        raise NotImplementedError  # use DPHandler instead

    def _compute_sanitized_gradients(self, loss: torch.Tensor):
        px_grads = loss  # TODO: per_example_gradients.
        # now assumes batch_size = 1
        add_noise = True
        sanitized_grads = self.sanitizer.sanitize(
            px_grads, self.sigma,
            num_examples=self.batches_per_lot * px_grads.size()[0],
            add_noise=add_noise
        )
        return sanitized_grads

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                """
                modified the line below
                old: d_p = p.grad.data
                """
                d_p = self._compute_sanitized_gradients(p.grad.data)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                p.data.add_(-group['lr'], d_p)

        return loss
