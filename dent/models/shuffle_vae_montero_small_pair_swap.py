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

        # self.eencoder = dent.models.encoder.montero_small.Encoder(
        #     img_size, self.latent_dim, dist_nparams=self.dist_nparams)

        
        self.mean_projector = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.logvar_projector = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # self.mmean_projector = nn.Sequential(
        #     nn.Linear(self.latent_dim, self.latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )

        # self.llogvar_projector = nn.Sequential(
        #     nn.Linear(self.latent_dim, self.latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )

        # 修改预测器为多层感知机，输出每个位置的变化概率
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.reset_parameters()

        # for p in self.mmean_projector.parameters():
        #     p.requires_grad_(False)
        # for p in self.llogvar_projector.parameters():
        #     p.requires_grad_(False)
        # for p in self.eencoder.parameters():
        #     p.requires_grad_(False)

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

        # me, md = self.momentum, self.momentum
        # with torch.no_grad():
        #     for p1,p2 in zip(self.encoder.parameters(), self.eencoder.parameters()):
        #         p2.data = md*p2.data + (1-md)*p1.data 
            # for p1,p2 in zip(self.mean_projector.parameters(), self.mmean_projector.parameters()):
            #     p2.data = md*p2.data + (1-md)*p1.data 
            # for p1,p2 in zip(self.logvar_projector.parameters(), self.llogvar_projector.parameters()):
            #     p2.data = md*p2.data + (1-md)*p1.data 

        stats_qzx = self.encoder(x)['stats_qzx']
        project_stats_qzx = torch.stack([self.mean_projector(stats_qzx[:,:,0]), self.logvar_projector(stats_qzx[:,:,1])], dim=2)

        # swap
        shuffle_stats_qzx, shuffle_history = self.pair_and_swap_latent(stats_qzx)
        pair_indices, dims_to_swap = shuffle_history
        
        # 获取每个样本对应的交换维度（向量化操作）
        batch_size = stats_qzx.shape[0]
        batch_indices = torch.arange(batch_size, device=stats_qzx.device)
        mask_first = pair_indices[:, 0].unsqueeze(1) == batch_indices
        mask_second = pair_indices[:, 1].unsqueeze(1) == batch_indices
        mask_combined = (mask_first | mask_second).any(dim=0)
        
        latent_shuffled_idx = torch.full((batch_size,), -1, device=stats_qzx.device)
        latent_shuffled_idx[pair_indices.flatten()] = dims_to_swap.repeat_interleave(2)
        latent_shuffled_idx = latent_shuffled_idx.tolist()
        
        # 对于未交换的样本随机选择维度，对于交换的样本选择非交换维度（向量化操作）
        unswapped_mask = torch.tensor(latent_shuffled_idx, device=stats_qzx.device) == -1
        random_dims = torch.randint(0, self.latent_dim, (batch_size,), device=stats_qzx.device)
        
        # 为交换的样本生成有效的随机维度
        swapped_dims = torch.tensor(latent_shuffled_idx, device=stats_qzx.device)
        valid_dims = torch.arange(self.latent_dim, device=stats_qzx.device).unsqueeze(0).expand(batch_size, -1)
        valid_mask = valid_dims != swapped_dims.unsqueeze(1)
        
        # 使用gather操作代替masked_select来提高效率
        random_indices = torch.randint(0, self.latent_dim - 1, (batch_size,), device=stats_qzx.device)
        valid_dims_masked = valid_dims[valid_mask].view(batch_size, -1)
        swapped_random_dims = valid_dims_masked[torch.arange(batch_size, device=stats_qzx.device), random_indices]
        
        latent_unshuffled_idx = torch.where(unswapped_mask, random_dims, swapped_random_dims)
        shuffle_project_stats_qzx = torch.stack([self.mean_projector(shuffle_stats_qzx[:,:,0]), self.logvar_projector(shuffle_stats_qzx[:,:,1])], dim=2)
        # de-shuffle
        deshuffle_project_stats_qzx = self.undo_pair_and_swap_latent(shuffle_project_stats_qzx, shuffle_history)
        # deshuffle_project_stats_qzx_pair = self.predictor(deshuffle_project_stats_qzx).reshape(deshuffle_project_stats_qzx.shape[0], -1, 2)


        samples_qzx = self.reparameterize(*stats_qzx.unbind(-1))['samples_qzx']
        reconstructions = self.decoder(samples_qzx.detach())['reconstructions']

        project_samples_qzx = self.reparameterize(*stats_qzx.unbind(-1))['samples_qzx']
        shuffle_project_samples_qzx = self.reparameterize(*shuffle_stats_qzx.unbind(-1))['samples_qzx']

        # shuffle_logit = self.predictor(torch.cat([project_samples_qzx, shuffle_project_samples_qzx], dim=-1))
        # shuffle_logit = self.predictor(project_samples_qzx - shuffle_project_samples_qzx)
        # 使用向量差来计算变化
        diff = torch.abs(project_samples_qzx - shuffle_project_samples_qzx)
        shuffle_logit = self.predictor(diff)  # 直接预测变化位置的概率分布

        return {
            'reconstructions': reconstructions, 
            'stats_qzx': stats_qzx,
            'samples_qzx': samples_qzx,
            'stats_qzx_list': (project_stats_qzx, deshuffle_project_stats_qzx),
            'stats_qzx_list_pair': None, # (project_stats_qzx_pair, deshuffle_project_stats_qzx_pair),
            'idx': (latent_shuffled_idx, latent_unshuffled_idx),
            'shuffle_logit': shuffle_logit
            }
    
    def pair_and_swap_latent(self, x):
        batch_size, latent_dim, feature_dim = x.shape
        device = x.device
        
        pair_indices = torch.randperm(batch_size, device=device).view(-1, 2)
        dims_to_swap = torch.randint(0, latent_dim, (len(pair_indices),), device=device)
        
        swap_mask = torch.zeros(batch_size, latent_dim, feature_dim, dtype=torch.bool, device=device)
        swap_mask[pair_indices[:, 0], dims_to_swap] = True
        swap_mask[pair_indices[:, 1], dims_to_swap] = True
        keep_mask = ~swap_mask
        
        x_swapped = torch.where(keep_mask, x, 
                                x[pair_indices.flatten()].view(batch_size, latent_dim, feature_dim)[pair_indices.argsort().view(-1)])
        
        shuffle_history = (pair_indices, dims_to_swap)
        
        return x_swapped, shuffle_history

    def undo_pair_and_swap_latent(self, x, shuffle_history):
        pair_indices, dims_to_swap = shuffle_history
        batch_size, latent_dim, feature_dim = x.shape
        device = x.device
        
        swap_mask = torch.zeros(batch_size, latent_dim, feature_dim, dtype=torch.bool, device=device)
        swap_mask[pair_indices[:, 0], dims_to_swap] = True
        swap_mask[pair_indices[:, 1], dims_to_swap] = True
        keep_mask = ~swap_mask
        
        x_unswapped = torch.where(keep_mask, x, 
                                  x[pair_indices.flatten()].view(batch_size, latent_dim, feature_dim)[pair_indices.argsort().view(-1)])
        
        return x_unswapped
    

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