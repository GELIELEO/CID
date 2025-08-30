# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
from torch import nn

from dent.models.encoder.misc.sysbinder import SysBinder

class Encoder(nn.Module):

    def __init__(self, img_size, latent_dim=10, dist_nparams=2):
        r"""Small Encoder as utilised in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        dist_nparams : int
            number of distribution statistics to return

        References:
            [1] Montero et al. "Lost in Latent Space: Disentangled Models and 
            the Challenge of Combinatorial Generalisation."
        """
        super(Encoder, self).__init__()

        # Layer parameters
        kernel_size = 4
        self.latent_dim = latent_dim
        self.img_size = img_size
        n_chan = self.img_size[0]

        assert_str = "This architecture requires 64x64 inputs."
        assert self.img_size[-2] == self.img_size[-1] == 64, assert_str

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, 32, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(32, 32, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(32, 64, kernel_size, **cnn_kwargs)
        self.conv4 = nn.Conv2d(64, 64, kernel_size, **cnn_kwargs)
        self.conv5 = nn.Conv2d(64, 128, kernel_size, **cnn_kwargs)

        self.binder = SysBinder(
            num_iterations=3,
            num_slots=1,
            input_size=128,
            slot_size=latent_dim,
            mlp_hidden_size=25,
            num_prototypes=5,
            num_blocks=5
        )


    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))

        print(x.shape)

        b,c,h,w = x.shape
        x = x.reshape(b,c,-1).permute(0,2,1)
        x, attn = self.binder(x)
        x = x.view((batch_size, -1))        

        return {'latent_z': x}


if __name__=='__main__':
    encoder = Encoder((1, 64, 64), latent_dim=10)
    fake_input = torch.randn((1, 1, 64, 64))
    fake_output = encoder(fake_input)