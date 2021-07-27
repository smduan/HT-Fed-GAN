#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:hty
@file: vmserver.py
@time: 2021/05/16
"""

import torch
import syft as sy
import numpy as np


class Server(object):

    def __init__(self, *, n_components=1, n_samples=None, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='random',
                 weight_concentration_prior_type='dirichlet_distribution',
                 weight_concentration_prior=None,
                 mean_precision_prior=1, mean_prior=None,
                 degrees_of_freedom_prior=1, covariance_prior=None,
                 random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10,rn=None, hook=None):
        self.n_components = n_components
        self.n_samples = n_samples
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.covariance_type = covariance_type
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior_ = weight_concentration_prior
        self.mean_precision_prior_ = mean_precision_prior
        self.means_prior_ = mean_prior
        self.degrees_of_freedom_prior_ = degrees_of_freedom_prior
        self.covariance_prior_ = covariance_prior
        self.clients = []
        self.data = {}
        self.server = sy.VirtualWorker(hook,id="server")
        self.hook = hook
        # only for test, usually it is randomly generated.
        self.rn = rn

    def connect_client(self,Client):
        self.clients.append(Client.client)
        self.data[Client.client] = Client.data_ptr

    def connect_cw(self,cw):
        self.crypto_provider = cw

    def my_div(self, one, another):
        dividend = one.get().float_precision()
        divisor = another.get().float_precision()
        result = dividend / (divisor + 10 * np.finfo(float).eps)
        return result

    def initialize_rn(self, random_state):
        """
        server randomly generate resp 
        :param random_state:
        :return:initialized rn
        """
        if self.init_params == 'random' and self.rn is None:
            resp = random_state.rand(self.n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        elif self.init_params == 'random' and self.rn is not None:
            resp =self.rn
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)
        self.resp = resp

    def init_nk(self):
        """
        calculate Nk_0
        :return:
        """

        nk = self.resp.sum(axis=0) + 10 * np.finfo(self.resp.dtype).eps
        nk_tensor = torch.from_numpy(nk)
        # nk_tensor_inverse = torch.from_numpy(nk_inverse).reshape((3,1))
        nk_share = nk_tensor.fix_precision().share(self.server, self.crypto_provider,crypto_provider = self.crypto_provider)
        # nk_inverse_share = nk_tensor_inverse.fix_precision().share(self.server, self.crypto_provider,crypto_provider = self.crypto_provider)

        # return nk_share, nk_inverse_share
        return nk_share

    def calcu_nk(self, resp_list):
        nk_server = 0
        for resp in resp_list:
            nk_server += torch.sum(resp, dim=0)
        # nk_server += 10 * np.finfo(float).eps
        nk = nk_server.get().float_precision()
        nk += 0.001
        nk = nk.fix_precision().share(self.server, self.crypto_provider,crypto_provider=self.crypto_provider)

        return nk

    def calcu_x_bar(self,x_sum_list, nk):
        
        x_sum_server = 0
        for x_sum in x_sum_list:
            x_sum_server = x_sum_server + x_sum

        # x_sum = x_sum_server.get().float_precision()
        # x_sum_crypto = x_sum.fix_precision().share(self.clients[1], self.clients[0],crypto_provider=self.clients[0])

        nk_ = nk.reshape((self.n_components,1))

        x_bar = self.my_div(x_sum_server, nk_)

        xk = x_bar.fix_precision().share(self.server, self.crypto_provider,crypto_provider=self.crypto_provider)

        # x_bar = x_sum_server / nk_
        # x_bar = x_bar.get().float_precision()
        # xk = x_bar.fix_precision().share(self.server, self.crypto_provider,crypto_provider=self.crypto_provider)

        # return xk,x_sum_server
        return xk

    def calcu_means_prior(self, sum):
        n = self.n_samples
        sum = torch.cat(sum, dim=1)
        means_prior_ = torch.sum(sum, dim=1)/n
        self.means_prior_ = means_prior_

    def calcu_covariances_prior(self, covariances):
        n = self.n_samples
        covariances = torch.stack(covariances,2)
        covariances_prior_ = torch.sum(covariances, dim=2) / (n-1)
        self.covariance_prior_ = covariances_prior_

    def send_resp(self):
        """
        send resp to clients,this step is only used when initializing
        :return:
        """

        resp_list = []
        current_start = 0

        for client in self.clients:
            x = self.data[client]
            current_n_sample = len(x)
            resp_tensor = torch.from_numpy(self.resp[current_start:current_start + current_n_sample]).tag("resp")
            resp_tensor_client = resp_tensor.send(client)
            resp_list.append(resp_tensor_client)
            current_start += current_n_sample

        return resp_list

    def send_params_prior(self):
        """
        this step is only used when initializing
        and initilized all params with '_prior'
        :return: the dict of initialized params
        """
        # calculate mean_prior and covariance_prior firstly
        params_prior = dict()
        params_prior["weight_concentration_prior_"] = torch.tensor(self.weight_concentration_prior_)
        params_prior["means_prior_"] = torch.tensor(self.means_prior_)
        params_prior["mean_precision_prior_"] = torch.tensor(self.mean_precision_prior_)
        params_prior["covariance_prior_"] = torch.tensor(self.covariance_prior_)
        params_prior["degrees_of_freedom_prior_"] = torch.tensor(self.degrees_of_freedom_prior_)

        return params_prior

    def estimate_params(self, nk, xk):

        # calculate α
        weight_concentration_ = self.weight_concentration_prior_ + nk
        # calculate β
        mean_precision_ = self.mean_precision_prior_ + nk

        # calculate m
        x_sum = nk.reshape((-1,1)) * xk

        n = torch.tensor([self.n_samples]).fix_precision().share(self.server, self.crypto_provider,
                                                                                 crypto_provider=self.crypto_provider)
        means_prior_ = self.means_prior_
        means_prior_ = means_prior_.fix_precision().share(self.server, self.crypto_provider,
                                                                                 crypto_provider=self.crypto_provider)
        means_ = self.my_div(self.mean_precision_prior_ * means_prior_ + x_sum, mean_precision_.reshape(-1,1))
        means_ = means_.fix_precision().share(self.server, self.crypto_provider,crypto_provider=self.crypto_provider)

        # calculate v
        degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        # because pysyft can't realise decrypting a shared tensor on multiple clients
        # so we decrypt params then send them instead decrypt after send
        params_dict = {
            "Nk": nk.get().float_precision(),
            "xk": xk.get().float_precision(),
            "weight_concentration_":weight_concentration_.get().float_precision(),
            "mean_precision_":mean_precision_.get().float_precision(),
            "means_":means_.get().float_precision(),
            "degrees_of_freedom_":degrees_of_freedom_.get().float_precision()
        }

        return params_dict

    def m_step(self, x_sum_list, resp_list):
        """
        merge xk and nk from clients，then estimate new params
        :param x_sum_list:xk from clients
        :param resp_list:rn from clients
        :return: merged params
        """
        nk = self.calcu_nk(resp_list)
        xk = self.calcu_x_bar(x_sum_list, nk)
        params = self.estimate_params(nk, xk)

        return nk, xk, params

    def aggregation_and_convergence_check(self, params, sk_list, covariances_):
        # merge sk
        sk_shape = sk_list[0].shape
        sk = torch.from_numpy(np.zeros(sk_shape,dtype=float))
        sk_server = sk.fix_precision().share(self.server, self.crypto_provider,crypto_provider=self.crypto_provider)

        for s in sk_list:
            sk_server = sk_server + s
        for i in range(sk_shape[0]):
            # set reg_covar on diagonal of sk
            for j in range(sk_shape[1]):
                sk_server[i][j][j] += self.reg_covar

        # merge lk which don't need to be encrypted
        # merge Wk
        nk = params["Nk"]
        degrees_of_freedom_ = params["degrees_of_freedom_"]
        nk_server = nk.fix_precision().share(self.server, self.crypto_provider,crypto_provider=self.crypto_provider)
        degrees_of_freedom_server = degrees_of_freedom_.fix_precision().share(self.server, self.crypto_provider,crypto_provider=self.crypto_provider)
        for k in range(self.n_components):
            covariances_[k] += nk_server[k] * sk_server[k]
        # covariances_ /= degrees_of_freedom_server[:, np.newaxis, np.newaxis]
        # use self.my_div solve some problem of pysyft
        covariances_ = self.my_div(covariances_, degrees_of_freedom_server[:, np.newaxis, np.newaxis])

        sk = sk_server.get().float_precision()
        return sk,covariances_