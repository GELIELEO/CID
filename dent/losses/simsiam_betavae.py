# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fastargs.decorators import param
import torch
import math

from lightly.loss import BarlowTwinsLoss

import dent.losses.baseloss
import torch.utils
from .utils import _reconstruction_loss, _kl_normal_loss
from .modules.kld import kl_divergence_two_gaussian

class Loss(dent.losses.baseloss.BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """

    @param('betavae.beta')
    @param('betavae.log_components')
    @param('betatcvae.is_mss')
    @param('rect.coeff')
    def __init__(self, n_data, beta, log_components, is_mss=True, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.log_components = log_components
        self.is_mss = is_mss
        self.n_data = n_data
        self.coeff = kwargs['coeff']

        self.barlow_loss = BarlowTwinsLoss(lambda_param=1.0)

    def __call__(self, data, reconstructions, stats_qzx, stats_qzx_list, is_train, **kwargs):   
        self._pre_call(is_train)
        
        log_data = {}
        rec_loss = _reconstruction_loss(data, reconstructions, distribution=self.rec_dist)
        log_data['rec_loss'] = rec_loss.item()

        kl_loss = _kl_normal_loss(*stats_qzx.unbind(-1), return_components=True)
        if self.log_components:
            log_data.update(
                {f'kl_loss_{i}': value.item() for i, value in enumerate(kl_loss)})
        kl_loss = kl_loss.sum()
        log_data['kl_loss'] = kl_loss.item()

        alpha = 0.5
        deshuffle_kl_loss = self.bidirectional_kl(stats_qzx_list[0], stats_qzx_list[1], alpha) + self.bidirectional_kl(stats_qzx_list[2], stats_qzx_list[3], alpha)
        log_data['deshuffle_kl_loss'] = deshuffle_kl_loss.item()

        gamma = 0.5
        main_deshuffle_kl_loss = self.bidirectional_kl(stats_qzx, stats_qzx_list[0], gamma)
        log_data['main_deshuffle_kl_loss'] = main_deshuffle_kl_loss.item()

        barlow_loss = self.barlow_loss(stats_qzx.unbind(-1)[0], stats_qzx_list[0].unbind(-1)[0])
        log_data['barlow_loss'] = barlow_loss.item()

        loss = self.coeff*rec_loss + self.beta * kl_loss + 0.1*deshuffle_kl_loss
        log_data['loss'] = loss.item()

        return {'loss': loss, 'to_log': log_data}
    
    def bidirectional_kl(self, stats_qzx_1, stats_qzx_2, alpha):
        kl_loss = alpha*kl_divergence_two_gaussian(*stats_qzx_1.detach().unbind(-1), *stats_qzx_2.unbind(-1)).mean() \
            + (1-alpha)*kl_divergence_two_gaussian(*stats_qzx_2.unbind(-1), *stats_qzx_1.detach().unbind(-1)).mean()
        return kl_loss
