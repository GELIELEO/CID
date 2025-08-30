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

from dent.losses.modules.kl_matrix import compute_kl_divergence_matrix



class Model(nn.Module):
    @param('vae_locatello.latent_dim')
    @param('shuffled_betavae.shuffle_dims')
    @param('shuffled_betavae.momentum')
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
        
        self.shuffle_dims = kwargs['shuffle_dims']
        self.momentum = kwargs['momentum']
        
        self.model_name = 'vae_montero_small'

        self.eencoder = dent.models.encoder.montero_small.Encoder(
            img_size, self.latent_dim, dist_nparams=self.dist_nparams)
        self.ddecoder = dent.models.decoder.montero_small.Decoder(
            img_size, self.latent_dim)

        self.reset_parameters()

        for p in self.eencoder.parameters():
            p.requires_grad_(False)
        for p in self.ddecoder.parameters():
            p.requires_grad_(False)

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

        me, md = self.momentum, self.momentum
        with torch.no_grad():
            for p1,p2 in zip(self.encoder.parameters(), self.eencoder.parameters()):
                p2.data = me*p2.data + (1-me)*p1.data
            for p1,p2 in zip(self.decoder.parameters(), self.ddecoder.parameters()):
                p2.data = md*p2.data + (1-md)*p1.data 
        # self.eencoder.load_state_dict(self.encoder.state_dict())
        # self.ddecoder.load_state_dict(self.decoder.state_dict())

        # VAE
        # *torch.rand_like(x) if self.train else x
        # mask = (torch.rand_like(x)>0.5)*1
        # *mask+x*torch.rand_like(x)*(1-mask) if self.training else x
        stats_qzx = self.encoder(x)['stats_qzx']
        samples_qzx = self.reparameterize(*stats_qzx.unbind(-1))['samples_qzx']
        reconstructions = self.decoder(samples_qzx)['reconstructions']
        # stats_qzx_pair = self.predictor(stats_qzx).reshape(stats_qzx.shape[0], -1, 2)


        # shuffle
        number_of_latents_to_shuffle = self.shuffle_dims
        latent_tobe_shuffled = torch.randperm(self.latent_dim)
        latent_shuffled_idx = latent_tobe_shuffled[:number_of_latents_to_shuffle]
        latent_unshfflued_idx, _ = latent_tobe_shuffled[number_of_latents_to_shuffle:].sort()
        shuffle_stats_qzx, shuffle_history = self.shuffle_latent(stats_qzx, latent_shuffled_idx)

        shuffle_sample_qzx = self.reparameterize(*shuffle_stats_qzx.unbind(-1))['samples_qzx'] 
        shuffle_reconstructions = self.ddecoder(shuffle_sample_qzx)['reconstructions']
        shuffle_reconstructions_stats_qzx = self.eencoder(shuffle_reconstructions)['stats_qzx']
        
        # shuffle_reconstructions_stats_qzx = torch.stack([self.mean_mixer(shuffle_stats_qzx[:,:,0]), self.logvar_mixer(shuffle_stats_qzx[:,:,1])], dim=2)

        # de-shuffle
        # deshuffle_stats_qzx = torch.clone(shuffle_reconstructions_stats_qzx)
        deshuffle_stats_qzx = self.deshuffle_latent(shuffle_reconstructions_stats_qzx, latent_shuffled_idx, shuffle_history)
        # deshuffle_stats_qzx_pair = self.predictor(deshuffle_stats_qzx).reshape(deshuffle_stats_qzx.shape[0], -1, 2)

        # deshuffle_sample_qzx = self.reparameterize(*deshuffle_stats_qzx.unbind(-1))['samples_qzx'] 
        # deshuffle_reconstructions = self.ddecoder(deshuffle_sample_qzx)['reconstructions']

        if not self.training:
            swap_results = []
            latent_to_swap = torch.linspace(0, self.latent_dim-1, self.latent_dim, dtype=torch.long)
            for i in latent_to_swap:
                swap_stats_qzx, _, swap_stats_qzx_ = self.swap_latent(stats_qzx.clone(), [i,])
                swap_results.append(
                    self.ddecoder(self.reparameterize(*swap_stats_qzx.unbind(-1))['samples_qzx'])['reconstructions']
                    )
                swap_results.append(
                    self.ddecoder(self.reparameterize(*swap_stats_qzx_.unbind(-1))['samples_qzx'])['reconstructions']
                    )
            shuffle_reconstructions = torch.stack(swap_results, dim=1)

        return {
            'reconstructions': reconstructions, 
            'stats_qzx': stats_qzx,
            'samples_qzx': samples_qzx,
            'deshuffle_stats_qzx': deshuffle_stats_qzx,
            'stats_qzx_list': (stats_qzx, deshuffle_stats_qzx),
            'stats_qzx_list_pair': None,# (stats_qzx_pair, deshuffle_stats_qzx_pair),
            'shuffle_rect_stats_qzx_list': (shuffle_stats_qzx, shuffle_reconstructions_stats_qzx),
            'idx': (latent_shuffled_idx, latent_unshfflued_idx),
            'shuffle_reconstructions': shuffle_reconstructions,
            }

    def shuffle_latent(self, x, latent_shuffled_idx):
        shuffle_history = []
        for i in latent_shuffled_idx:
            sample_to_shuffle_idx = torch.randperm(x.shape[0])
            # x.permute(1,0,2)[i] = x.permute(1,0,2)[i][sample_to_shuffle_idx]
            x = torch.cat([x[:,:i], x[:,i:i+1][sample_to_shuffle_idx], x[:,i+1:]], dim=1)
            shuffle_history.append(sample_to_shuffle_idx)
        return x, shuffle_history

    def deshuffle_latent(self, x, latent_shuffled_idx, shuffle_history):
        for i, j in enumerate(latent_shuffled_idx):
            sample_idx, indices = torch.sort(shuffle_history[i])
            # x.permute(1,0,2)[j] = x.permute(1,0,2)[j][indices]
            x = torch.cat([x[:,:j], x[:,j:j+1][indices], x[:,j+1:]], dim=1)
        return x

    def swap_latent(self, x, latent_shuffled_idx):
        shuffle_history = []
        x_ = x.clone()
        for i in latent_shuffled_idx:
            sample_to_shuffle_idx = torch.linspace(x.shape[0]-1, 0, x.shape[0], dtype=torch.long) # head-tail swap
            # x.permute(1,0,2)[i] = x.permute(1,0,2)[i][sample_to_shuffle_idx]
            x = torch.cat([x[:,:i], x[:,i:i+1][sample_to_shuffle_idx], x[:,i+1:]], dim=1)
            x_ = torch.cat([x_[:,:i][sample_to_shuffle_idx], x_[:,i:i+1], x_[:,i+1:][sample_to_shuffle_idx]], dim=1)
            shuffle_history.append(sample_to_shuffle_idx)
        return x, shuffle_history, x_
    
    def replace_latent(self, x, latent_shuffled_idx):
        shuffle_history = []
        for i in latent_shuffled_idx:
            kl_matrix = compute_kl_divergence_matrix(x)
            sample_to_shuffle_idx = kl_matrix.argmax(dim=1)
            x = torch.cat([x[:,:i], x[:,i:i+1][sample_to_shuffle_idx], x[:,i+1:]], dim=1)
            shuffle_history.append(sample_to_shuffle_idx)
        return x, shuffle_history
    
    

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