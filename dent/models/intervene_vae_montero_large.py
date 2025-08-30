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

import dent.utils.initialization
import dent.models.encoder.montero_large
import dent.models.decoder.montero_large


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
        self.encoder = dent.models.encoder.montero_large.Encoder(
            img_size, self.latent_dim, dist_nparams=self.dist_nparams)
        self.decoder = dent.models.decoder.montero_large.Decoder(
            img_size, self.latent_dim)
        self.model_name = 'vae_montero_large'
        self.reset_parameters()

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
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return {'samples_qzx': mean + std * eps}
        else:
            # Reconstruction mode
            return {'samples_qzx': mean}

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        stats_qzx = self.encoder(x)['stats_qzx']
        samples_qzx = self.reparameterize(*stats_qzx.unbind(-1))['samples_qzx']
        reconstructions = self.decoder(samples_qzx)['reconstructions']

        # rand_idx = torch.randint(0, self.latent_dim, (x.shape[0],))
        # intervene_pos = F.one_hot(rand_idx, self.latent_dim).to(x.dtype).to(x.device).unsqueeze(-1)
        # shuffled_idx = torch.randperm(x.shape[0])
        # intervene_stats_qzx = stats_qzx*(1-intervene_pos)+stats_qzx[shuffled_idx]*intervene_pos 
        
        latent_to_shuffled_idx = torch.randperm(self.latent_dim)[:1]
        # intervene_stats_qzx = torch.stack(
        #     [
        #         stats_qzx.permute(1,0,2)[i][torch.randperm(x.shape[0])] if i in latent_to_shuffled_idx else stats_qzx.permute(1,0,2)[i]
        #         for i in range(self.latent_dim)
        #     ]
        # ).permute(1,0,2)
        intervene_stats_qzx = torch.clone(stats_qzx)
        for i in latent_to_shuffled_idx:
            intervene_stats_qzx.permute(1,0,2)[i] = stats_qzx.permute(1,0,2)[i][torch.randperm(x.shape[0])]
        
        intervene_sample_qzx = self.reparameterize(*intervene_stats_qzx.unbind(-1))['samples_qzx'] 
        intervene_reconstructions = self.decoder(intervene_sample_qzx)['reconstructions']
        intervene_reconstructions_stats_qzx = self.encoder(intervene_reconstructions)['stats_qzx']

        return {
            'reconstructions': reconstructions, 
            'stats_qzx': stats_qzx, 
            'samples_qzx': samples_qzx,
            'invert_stats_qzx': (intervene_stats_qzx, intervene_reconstructions_stats_qzx),
            'intervene_reconstructions': intervene_reconstructions
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
