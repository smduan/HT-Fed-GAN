#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:hty
@file: vnclient.py
@time: 2021/05/16
"""

import torch
import syft as sy
import pandas as pd
import numpy as np

from scipy import linalg

class Client(object):
    '''
    some paramters are needed as model initialise：
    name：like "alice"
    path：like "alice_data.csv"
    usecols：like [1,2,3]
    '''
    def __init__(self, name, path, usecols, hook):
        # create vm
        self.client = sy.VirtualWorker(hook,id=name)

        # prepare data
        file = pd.read_csv(path, usecols=usecols)
        data_tensor = torch.from_numpy(file.values).double()
        self.data_ptr = data_tensor.send(self.client)

    def connect_server(self, server):
        self.server = server

    def connect_other_client(self, clients):
        self.clients = clients

    def connect_cw(self, cw):
        self.crypto_provider = cw

    def get_params_prior(self, params_prior):
        self.weight_concentration_prior_ = params_prior["weight_concentration_prior_"]
        self.means_prior_ = params_prior["means_prior_"]
        self.mean_precision_prior_ = params_prior["mean_precision_prior_"]
        self.covariance_prior_ = params_prior["covariance_prior_"]
        self.degrees_of_freedom_prior_ = params_prior["degrees_of_freedom_prior_"]

    def calcu_x_sum(self):

        if len(self.client.search(["resp"])) != 1:
            raise ValueError("发现多个resp！")
        resp_ptr = self.client.search(["resp"])[0]

        x = self.data_ptr

        x_sum = torch.mm(resp_ptr.t(), x)
        x_sum = x_sum.get()
        share_node = self.clients + [self.server]
        x_sum = x_sum.fix_precision().share(self.server,self.crypto_provider,crypto_provider=self.crypto_provider)

        return x_sum

    def calcu_means_prior(self):
        sum = torch.sum(self.data_ptr,dim=0)
        return sum.get()

    def calcu_covariances_prior(self, means_prior_):
        _, n_feature = self.data_ptr.shape

        means_prior_ = means_prior_.send(self.client)

        c = np.atleast_2d(np.zeros((n_feature, n_feature), dtype=float))
        covariances_prior_client = torch.from_numpy(c).send(self.client)
        for i in range(n_feature):
            for j in range(n_feature):
                mean_prior_i = means_prior_[i]
                mean_prior_j = means_prior_[j]
                covariances_prior_client[i, j] = torch.sum(
                    torch.mul(self.data_ptr[:, i] - mean_prior_i, self.data_ptr[:, j] - mean_prior_j))
                covariances_prior_client[j, i] = torch.sum(
                    torch.mul(self.data_ptr[:, i] - mean_prior_i, self.data_ptr[:, j] - mean_prior_j))

        return covariances_prior_client.get()

    def m_step(self,params_dict,if_init=False, log_resp = None):
        # get param resp 
        if len(self.client.search(["resp"])) == 1:
            resp = self.client.search(["resp"])[0]
        else:
            raise ValueError("发现多个resp！")

        # get param xk,nk which was encrypted
        xk = params_dict["xk"]
        nk = params_dict["Nk"]

        xk_client = xk.send(self.client)
        nk_client = nk.send(self.client)
        mean_precision_ = params_dict["mean_precision_"]
        # calculate sk
        n_components, n_features = xk_client.shape
        sk = np.empty((n_components, n_features, n_features))
        sk = torch.from_numpy(sk).send(self.client)

        for k in range(n_components):
            means_conponent = xk_client[k]
            # diff = torch.sub(self.data_ptr, means_conponent)
            diff = self.data_ptr - means_conponent.double()
            tmp = (resp[:, k] * diff.t()).get().send(self.client)
            sk[k] = torch.div(torch.mm(tmp, diff), nk_client[k])

        sk = sk.get()
        sk_share = sk.fix_precision().share(self.server, self.crypto_provider, crypto_provider=self.crypto_provider)

        # calculate lk in client 0
        if if_init == True:
            if(self.client == self.clients[0]):
                # log_det_precisions_chol_
                precisions_cholesky_ = self.precision_cholesky_.reshape(n_components, -1)[:, ::n_features + 1]
                precisions_cholesky_ = precisions_cholesky_.tag("temp")
                precisions_cholesky_client = precisions_cholesky_.send(self.client)
                log_det_client = torch.sum(torch.log(precisions_cholesky_client), 1)
                degrees_of_freedom_client = self.degrees_of_freedom_.send(self.client)
                log_det_precisions_chol_client = torch.sub(log_det_client,.5 * n_features * torch.log(degrees_of_freedom_client))
                
                # log_wishart
                tmp = torch.from_numpy(np.arange(n_features)[:, np.newaxis]).send(self.client)
                # _log_wishart_norm_client = -(degrees_of_freedom_client * log_det_precisions_chol_client +
                #                       degrees_of_freedom_client * n_features * .5 * np.log(2) +
                #                       torch.sum(torch.lgamma(.5 * (torch.sub(degrees_of_freedom_client, tmp))), dim=0))
                tmp1 = degrees_of_freedom_client * log_det_precisions_chol_client
                tmp2 = degrees_of_freedom_client * n_features * .5 * np.log(2)
                tmp3 = torch.sum(torch.lgamma(.5 * (torch.sub(degrees_of_freedom_client, tmp))), dim=0)
                tmp1 = tmp1.get().send(self.client)
                tmp2 = tmp2.get().send(self.client)
                tmp3 = tmp3.get().send(self.client)

                tmp4 = -tmp1.sub(tmp2)
                # tmp4 = -tmp1-tmp2
                _log_wishart_norm_client = tmp4.sub(tmp3)
                log_wishart_client = torch.sum(_log_wishart_norm_client)

                # log_norm_weight
                weight_concentration_ = self.weight_concentration_
                weight_concentration_client = weight_concentration_.send(self.client)
                tmp1 = torch.lgamma(torch.sum(weight_concentration_client))
                tmp2 = torch.sum(torch.lgamma(weight_concentration_client))
                log_norm_weight = tmp1 - tmp2

                # sum
                sum = 0
                for log_rn in log_resp:
                    # tmp = torch.sum(torch.exp(log_rn) * log_rn).fix_precision().share(self.workers[0],self.workers[1],self.workers[2]).get()
                    # tmp = torch.sum(torch.exp(log_rn) * log_rn).get().float_precision()
                    log_rn = log_rn.get().float_precision().send(self.client)
                    tmp = torch.sum(torch.exp(log_rn) * log_rn)
                    sum = sum + tmp
                # sum = sum.send(self.client)

                tmp1 = sum + log_wishart_client
                # tmp2 = .5 * n_features * np.sum(np.log(self.mean_precision_))
                mean_precision_client = mean_precision_.send(self.client)
                tmp2 = .5 * n_features * torch.sum(torch.log(mean_precision_client))
                result = (-tmp1 - tmp2)
                low_bound = result + log_norm_weight
            else:
                low_bound = None
        else:
            low_bound = None

        # calculate Wk in client 1
        if(self.client == self.clients[1]):
            # initial Wk equl 0
            covariances_ = np.zeros((n_components, n_features, n_features))
            covariances_ = torch.from_numpy(covariances_).send(self.client)

            means_prior_client = self.means_prior_.send(self.client) # m0
            mean_precision_client = mean_precision_.send(self.client) # βk
            covariance_prior_client = self.covariance_prior_.send(self.client) # W0
            for k in range(n_components):
                diff = torch.sub(xk_client[k], means_prior_client)
                covariances_[k] = covariance_prior_client + nk_client[k] \
                                  / mean_precision_client[k] * (diff * diff).unsqueeze(0)
            covariances_ = covariances_.get()
            covariances_share = covariances_.fix_precision().share(self.server, self.crypto_provider,
                                                                   crypto_provider=self.crypto_provider)
        else:
            covariances_share = None

        resp.get()

        return sk_share, covariances_share, low_bound

    def _compute_precision_cholesky(self, covariances_):
        """Compute the Cholesky decomposition of the precisions.

        Parameters
        ----------
        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.

        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.

        Returns
        -------
        precisions_cholesky : array-like
            The cholesky decomposition of sample precisions of the current
            components. The shape depends of the covariance_type.
        """
        estimate_precision_error_message = (
            "Fitting the mixture model failed because some components have "
            "ill-defined empirical covariance (for instance caused by singleton "
            "or collapsed samples). Try to decrease the number of components, "
            "or increase reg_covar.")

        covariances = covariances_.numpy()
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,np.eye(n_features),lower=True).T
        return precisions_chol

    def _estimate_log_prob(self, X, precisions_cholesky_):
        # We remove `n_features * np.log(self.degrees_of_freedom_)` because
        # the precision matrix is normalized
        # log_gauss = (_estimate_log_gaussian_prob(
        #     X, self.means_, self.precisions_cholesky_, self.covariance_type) -
        #     .5 * self.n_feature_in * np.log(self.degrees_of_freedom_))

        n_samples, n_features = X.shape
        worker = X.location

        # log_det
        n_components, _, _ = precisions_cholesky_.shape
        precisions_chol = precisions_cholesky_.reshape(n_components, -1)[:, ::n_features + 1]
        precisions_chol = torch.from_numpy(precisions_chol).tag("temp")
        precisions_chol_client = precisions_chol.send(X.location)
        log_det_client = torch.sum(torch.log(precisions_chol_client), 1)

        # log_det = log_det_client.get()
        # log_prob pointer
        log_prob = np.empty((n_samples, n_components))
        log_prob = torch.from_numpy(log_prob).tag("temp")
        log_porb_client = log_prob.send(X.location)
        means_ = self.means_.double().tag("temp")
        # means_client = means_.send(X.location)
        precisions_chol_ = torch.from_numpy(precisions_cholesky_).tag("temp")
        # precisions_chol_client = precisions_chol_.send(X.location)
        for k, (mu, prec_chol) in enumerate(zip(means_, precisions_chol_)):
            # y = torch.mm(X, precisions_chol_client[k]) - torch.mm(means_client[k], precisions_chol_client[k])
            p = prec_chol.send(worker)
            m = mu.send(worker)
            # y = torch.sub(torch.mm(X, prec_chol), torch.mm(mu.reshape(1,-1), prec_chol))
            y = torch.sub(torch.mm(X, p), torch.mm(m.reshape(1,-1), p))

            log_porb_client[:, k] = torch.sum(y.pow(2), axis=1)

        # log_gauss which is a pointer to tensor
        # log_gauss_client = -.5 * (self.n_feature_in * np.log(2 * np.pi) + log_porb_client) + log_det_client
        log_gauss_client = torch.sub(log_det_client, .5 * (n_features * np.log(2 * np.pi) + log_porb_client))
        # log_gauss = log_gauss_client.get().numpy()

        # log_lambda = self.n_feature_in * np.log(2.) + np.sum(digamma(
        #     .5 * (self.degrees_of_freedom_ -
        #           np.arange(0, self.n_feature_in)[:, np.newaxis])), 0)

        # log_lambda
        degrees_of_freedom_client = self.degrees_of_freedom_.send(worker)
        temp = torch.from_numpy(np.arange(0, n_features)[:, np.newaxis]).tag("temp").send(worker)
        log_lambda_client = torch.add(n_features * np.log(2.), torch.sum(torch.digamma(.5 * (torch.sub(degrees_of_freedom_client, temp))), 0))

        mean_precision_client = self.mean_precision_.send(worker)

        # return log_gauss + .5 * (log_lambda_ - self.n_feature_in / self.mean_precision_) - .5 * n_features * np.log(self.degrees_of_freedom_)
        # return log_gauss_client + .5 * (log_lambda_client - self.n_feature_in / mean_precision_client) - .5 * self.n_feature_in * torch.log(degrees_of_freedom_client)
        # return log_gauss_client + torch.sub(.5 * (torch.sub(log_lambda_client, torch.div(self.n_feature_in, mean_precision_client))), .5 * self.n_feature_in * torch.log(degrees_of_freedom_client))
        tmp1 = .5 * n_features * torch.log(degrees_of_freedom_client)
        tmp2 = .5 * (torch.sub(log_lambda_client, torch.div(n_features, mean_precision_client)))
        tmp3 = torch.sub(tmp1, tmp2)
        # result = tmp3.get() + log_gauss_client.get()
        result = tmp3 + log_gauss_client
        result = result.get()
        return result.send(X.location)

    def _estimate_log_weights(self, worker):
        weight_concentration_client = self.weight_concentration_.send(worker)
        return torch.sub(torch.digamma(weight_concentration_client),
                         torch.digamma(torch.sum(weight_concentration_client)))

    def _estimate_weighted_log_prob(self, X, precision_cholesky):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return self._estimate_log_prob(X, precision_cholesky) + self._estimate_log_weights(X.location)

    def set_params(self, params):
        self.weight_concentration_ = params["weight_concentration_"]
        self.mean_precision_ = params["mean_precision_"]
        self.means_ = params["means_"]
        self.degrees_of_freedom_ = params["degrees_of_freedom_"]

    def e_step(self, sk, covariances_, params):
        # calculate precisions_cholesky_ firstly
        self.set_params(params)
        precisions_cholesky_ = self._compute_precision_cholesky(covariances_)
        self.precision_cholesky_ = torch.from_numpy(precisions_cholesky_)
        weighted_log_prob_client = self._estimate_weighted_log_prob(self.data_ptr, precisions_cholesky_)

        # log(resp)
        log_prob_norm_client = torch.logsumexp(weighted_log_prob_client, axis=1)
        log_resp_client = torch.sub(weighted_log_prob_client, log_prob_norm_client[:, np.newaxis])

        # resp
        resp = torch.exp(log_resp_client).get().tag("resp")
        resp_client = resp.send(self.client)

        # xk
        xk_client = torch.mm(torch.exp(log_resp_client).t(), self.data_ptr)

        # encrption
        log_resp = log_resp_client.get()
        log_resp_share = log_resp.fix_precision().share(self.server, self.crypto_provider,
                                                        crypto_provider=self.crypto_provider)
        xk_client = xk_client.get()
        xk_client_share = xk_client.fix_precision().share(self.server,self.crypto_provider,crypto_provider=self.crypto_provider)

        resp_share = resp.fix_precision().share(self.server,self.crypto_provider,crypto_provider=self.crypto_provider)

        return xk_client_share, log_resp_share, resp_share, resp_client