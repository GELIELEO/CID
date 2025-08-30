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
import dent.models.encoder.montero_small
import dent.models.decoder.montero_small
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
        self.encoder = dent.models.encoder.montero_small.Encoder(
            img_size, self.latent_dim, dist_nparams=self.dist_nparams)
        self.decoder = dent.models.decoder.montero_small.Decoder(
            img_size, self.latent_dim)
        self.model_name = 'vae_montero_small'

        self.predictor = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        
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

        # VAE
        stats_qzx = self.encoder(x)['stats_qzx']
        samples_qzx = self.reparameterize(*stats_qzx.unbind(-1))['samples_qzx']
        reconstructions = self.decoder(samples_qzx)['reconstructions']
        
        number_of_latents_to_shuffle = 1
        
        # Branch-1
        latent_to_shuffled_idx = torch.randperm(self.latent_dim)[:number_of_latents_to_shuffle]
        # shuffle
        shuffle_stats_qzx = torch.clone(stats_qzx)
        shuffle_stats_qzx, shuffle_history = self.shuffle_latent(shuffle_stats_qzx, latent_to_shuffled_idx)
        # decode and encode
        shuffle_sample_qzx = self.reparameterize(*shuffle_stats_qzx.unbind(-1))['samples_qzx'] 
        shuffle_reconstructions = self.decoder(shuffle_sample_qzx)['reconstructions']
        shuffle_reconstructions_stats_qzx = self.encoder(shuffle_reconstructions)['stats_qzx']
        # de-shuffle
        deshuffle_stats_qzx = torch.clone(shuffle_reconstructions_stats_qzx)
        deshuffle_stats_qzx = self.deshuffle_latent(deshuffle_stats_qzx, latent_to_shuffled_idx, shuffle_history)
        deshuffle_stats_qzx_pair = self.predictor(deshuffle_stats_qzx)
        
        # Branch-2
        latent_to_shuffled_idx_other = torch.randperm(self.latent_dim)[:number_of_latents_to_shuffle]
        # shuffle
        shuffle_stats_qzx_other = torch.clone(stats_qzx)
        shuffle_stats_qzx_other, shuffle_history_other = self.shuffle_latent(shuffle_stats_qzx_other, latent_to_shuffled_idx_other)
        # decode and encode
        shuffle_sample_qzx_other = self.reparameterize(*shuffle_stats_qzx_other.unbind(-1))['samples_qzx'] 
        shuffle_reconstructions_other = self.decoder(shuffle_sample_qzx_other)['reconstructions']
        shuffle_reconstructions_stats_qzx_other = self.encoder(shuffle_reconstructions_other)['stats_qzx']
        # de-shuffle
        deshuffle_stats_qzx_other = torch.clone(shuffle_reconstructions_stats_qzx_other)
        deshuffle_stats_qzx_other = self.deshuffle_latent(deshuffle_stats_qzx_other, latent_to_shuffled_idx_other, shuffle_history_other)
        deshuffle_stats_qzx_other_pair = self.predictor(deshuffle_stats_qzx_other)

        return {
            'reconstructions': reconstructions, 
            'stats_qzx': stats_qzx,
            'samples_qzx': samples_qzx,
            'stats_qzx_list': (deshuffle_stats_qzx, deshuffle_stats_qzx_other_pair, deshuffle_stats_qzx_other, deshuffle_stats_qzx_pair),
            'shuffle_reconstructions': shuffle_reconstructions
            }

    def shuffle_latent(self, x, latent_to_shuffled_idx):
        shuffle_history = []
        for i in latent_to_shuffled_idx:
            sample_to_shuffle_idx = torch.randperm(x.shape[0])
            x.permute(1,0,2)[i] = x.permute(1,0,2)[i][sample_to_shuffle_idx]
            shuffle_history.append(sample_to_shuffle_idx)
        return x, shuffle_history

    def deshuffle_latent(self, x, latent_to_shuffled_idx, shuffle_history):
        for i, j in enumerate(latent_to_shuffled_idx):
            sample_idx, indices = torch.sort(shuffle_history[i])
            x.permute(1,0,2)[j] = x.permute(1,0,2)[j][indices]
        return x
    
    

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


if __name__ == '__main__':
    model = Model(img_size=(3, 64, 64), latent_dim=10)
    fake_data = torch.randn(10, 3, 64, 64)

    out = model(fake_data)