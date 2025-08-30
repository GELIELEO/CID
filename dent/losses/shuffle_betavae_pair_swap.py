# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fastargs.decorators import param
import torch
import math
import torch.nn.functional as F

from lightly.loss import BarlowTwinsLoss

from pytorch_msssim import ssim, ms_ssim

import dent.losses.baseloss
import torch.utils
from .utils import _reconstruction_loss, _kl_normal_loss
from .modules.kld import kl_divergence_two_gaussian, js_divergence_two_gaussian, wasserstein_distance_two_gaussian

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
    @param('rect.rect_coeff')
    @param('shuffled_betavae.shuffled_coeff')
    @param('shuffled_betavae.shuffled_target')
    def __init__(self, n_data, beta, log_components, is_mss=True, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.log_components = log_components
        self.is_mss = is_mss
        self.n_data = n_data
        self.rect_coeff = kwargs['rect_coeff']
        self.shuffled_coeff = kwargs['shuffled_coeff']
        self.shuffled_target = kwargs['shuffled_target']

        self.barlow_loss = BarlowTwinsLoss(lambda_param=0.005)

    def __call__(self, data, reconstructions, stats_qzx, stats_qzx_list, stats_qzx_list_pair, idx, shuffle_logit, is_train, **kwargs):   
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

        gamma = 0.5
        if self.shuffled_target==1:
            deshuffle_kl_loss = self.js_two(stats_qzx_list[0], stats_qzx_list[1].detach(), gamma) 
            # deshuffle_kl_loss = self.wa_two(stats_qzx, deshuffle_stats_qzx, gamma)
            # deshuffle_kl_loss = self.js_one(stats_qzx, stats_qzx_list_pair[1]) + self.js_one(deshuffle_stats_qzx, stats_qzx_list_pair[0])
            # deshuffle_kl_loss = self.bi_kl_two(stats_qzx, deshuffle_stats_qzx, gamma)
            # deshuffle_kl_loss = self.bi_kl_one(stats_qzx, deshuffle_stats_qzx, gamma) \
            #                   + self.bi_kl_one(deshuffle_stats_qzx, stats_qzx, gamma)

        
        log_data['deshuffle_kl_loss'] = deshuffle_kl_loss.item()

        barlow_loss = self.barlow_loss(stats_qzx_list[0].unbind(-1)[0], stats_qzx_list[1].unbind(-1)[0])
        log_data['barlow_loss'] = barlow_loss.item()

        labels = torch.zeros_like(shuffle_logit)
        labels[torch.arange(len(idx[0])), idx[0]] = 1
        classify_loss = torch.binary_cross_entropy_with_logits(shuffle_logit, labels).sum()/shuffle_logit.shape[0]
        log_data['classify_loss'] = classify_loss.item()

        loss =  1.0*classify_loss + self.rect_coeff * rec_loss #+ self.beta * kl_loss +
        log_data['loss'] = loss.item()

        return {'loss': loss, 'to_log': log_data}

    def bi_kl_one(self, stats_qzx_1, stats_qzx_2, alpha):
        kl_loss = alpha*kl_divergence_two_gaussian(*stats_qzx_1.detach().unbind(-1), *stats_qzx_2.unbind(-1)).mean() \
            + (1-alpha)*kl_divergence_two_gaussian(*stats_qzx_2.unbind(-1), *stats_qzx_1.detach().unbind(-1)).mean()
        return kl_loss
    
    def uni_kl_two(self, stats_qzx_1, stats_qzx_2, alpha):
        kl_loss = kl_divergence_two_gaussian(*stats_qzx_2.unbind(-1), *stats_qzx_1.unbind(-1)).mean()
        return kl_loss
    
    def uni_kl_two_another(self, stats_qzx_1, stats_qzx_2, alpha):
        kl_loss = kl_divergence_two_gaussian(*stats_qzx_1.unbind(-1), *stats_qzx_2.unbind(-1)).mean()
        return kl_loss

    def bi_kl_two(self, stats_qzx_1, stats_qzx_2, alpha):
        kl_loss = alpha*kl_divergence_two_gaussian(*stats_qzx_1.detach().unbind(-1), *stats_qzx_2.unbind(-1)).mean() \
            + (1-alpha)*kl_divergence_two_gaussian(*stats_qzx_2.detach().unbind(-1), *stats_qzx_1.unbind(-1)).mean()
        return kl_loss
    
    
    def js_two(self, stats_qzx_1, stats_qzx_2, alpha):
        return  js_divergence_two_gaussian(*stats_qzx_1.unbind(-1), *stats_qzx_2.unbind(-1)).mean()
    
    def wa_two(self, stats_qzx_1, stats_qzx_2, alpha):
        return wasserstein_distance_two_gaussian(*stats_qzx_1.unbind(-1), *stats_qzx_2.unbind(-1)).mean()