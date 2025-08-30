# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fastargs.decorators import param
import torch
from torch import nn, optim
from torch.nn import functional as F

import os


import dent.utils.initialization
import dent.models.encoder.montero_small
import dent.models.decoder.montero_small
from dent.models.module.neural_prior.model import SpikeModel
from dent.models.utils.likelihood import log_likelihood_normal

class Model(nn.Module):
    @param('vae_locatello.latent_dim')
    def __init__(self, img_size, latent_dim, **kwargs):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(Model, self).__init__()

        if list(img_size[1:]) not in [[64, 64]]:
            raise RuntimeError(
                "{} sized images not supported. Only ((None, 64, 64) supported. Build your own architecture or reshape images!"
                .format(img_size))

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.dist_nparams = 2
        self.encoder = dent.models.encoder.montero_small.Encoder(
            img_size, self.latent_dim, dist_nparams=self.dist_nparams)
        self.decoder = dent.models.decoder.montero_small.Decoder(
            img_size, self.latent_dim)
        self.model_name = 'vae_montero_small'
        self.reset_parameters()

        model_path = '/home/ibasaw/causal_ws/WorldModel/disentangling-correlated-factors/dent/models/module/neural_prior/checkpoints'
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        num_units = 10
        epoch = 654
        step = 20305
        self.neural_prior = SpikeModel(num_units)
        self.neural_prior.flow.flows[-1].quantize_enabled = False

        state_dict = torch.load(os.path.join(model_path, f'epoch={epoch}-step={step}.ckpt'), map_location=device)['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('model.', '')] = v
        self.neural_prior.load_state_dict(new_state_dict)
        for para in self.neural_prior.parameters():
            para.requires_grad = False
        print('>>> Loaded neural prior')

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)+1.0
            eps = torch.randn_like(mean)
            samples_qzx = mean + std * eps
            log_qzx = log_likelihood_normal(samples_qzx, mean, std)
            # print('log_qzx', log_qzx)
            return {'samples_qzx': samples_qzx, 'log_qzx': log_qzx}
        else:
            # Reconstruction mode
            std = torch.exp(0.5 * logvar)+1.0
            log_qzx = log_likelihood_normal(mean, mean, std)
            return {'samples_qzx': mean, 'log_qzx': log_qzx}

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        stats_qzx = self.encoder(x)['stats_qzx']
        sample = self.reparameterize(*stats_qzx.unbind(-1))
        samples_qzx = sample['samples_qzx']
        # log_qzx = sample['log_qzx']
        # log_pz = self.neural_prior.log_prob(samples_qzx)
        samples_qzx = self.neural_prior.inverse(samples_qzx)
        reconstructions = self.decoder(samples_qzx)['reconstructions']
        return {
            'reconstructions': reconstructions, 
            'stats_qzx': stats_qzx, 
            'samples_qzx': samples_qzx,
            # 'log_qzx': log_qzx,
            # 'log_pz': log_pz
            }

    def reset_parameters(self):
        self.apply(dent.utils.initialization.weights_init)

    def sample_qzx(self, x):
        """
        Returns a sample z from the latent distribution q(z|x).

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        stats_qzx = self.encoder(x)['stats_qzx']
        samples_qzx = self.reparameterize(*stats_qzx.unbind(-1))['samples_qzx']
        return samples_qzx
