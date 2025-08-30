import math
import os
from typing import Any, Tuple, Union

import timm
import torch
import torchvision
import torch.nn as nn


class ViTBlocks(torch.nn.Module):
    '''The main processing blocks of ViT. Excludes things like patch embedding and classificaton
    layer.

    Args:
        width: size of the feature dimension.
        depth: number of blocks in the network.
        end_norm: whether to end with LayerNorm or not.
    '''
    def __init__(
        self,
        width: int = 768,
        depth: int = 12,
        num_heads = 12,
        end_norm: bool = True,
    ):
        super().__init__()

        # transformer blocks from ViT
        ViT = timm.models.vision_transformer.VisionTransformer
        vit = ViT(embed_dim=width, depth=depth, num_heads=num_heads)
        self.layers = vit.blocks
        if end_norm:
            # final normalization
            self.layers.add_module('norm', vit.norm)

    def forward(self, x: torch.Tensor):
        return self.layers(x)



class Encoder(torch.nn.Module):
    '''Masked Autoencoder for visual representation learning.

    Args:
        image_size: (height, width) of the input images.
        patch_size: side length of a patch.
        keep: percentage of tokens to process in the encoder. (1 - keep) is the percentage of masked tokens.
        enc_width: width (feature dimension) of the encoder.
        dec_width: width (feature dimension) of the decoder. If a float, it is interpreted as a percentage
            of enc_width.
        enc_depth: depth (number of blocks) of the encoder
        dec_depth: depth (number of blocks) of the decoder
    '''
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (3, 224, 224),
        latent_dim: int = 10,
        patch_size: int = 32,
        keep: float = 0.25,
        enc_width: int = 768,
        enc_depth: int = 12,
        num_heads: int = 12,
        dist_nparams: int=2
    ):
        super().__init__()

        image_size = image_size[1:]
        self.latent_dim = latent_dim
        self.dist_nparams = dist_nparams

        assert image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size
        self.keep = keep
        self.n = (image_size[0] * image_size[1]) // patch_size**2  # number of patches

        self.enc_width = enc_width
        self.enc_depth = enc_depth

        # linear patch embedding
        self.embed_conv = torch.nn.Conv2d(3, enc_width, patch_size, patch_size)

        # mask token and position encoding
        self.register_buffer('pos_encoder', self.pos_encoding(self.n*2, enc_width).requires_grad_(False))

        # encoder
        self.encoder = ViTBlocks(width=enc_width, depth=enc_depth, num_heads=num_heads)

        # linear projection from enc_width to dec_width
        # self.project = torch.nn.Linear(enc_width, latent_dim)
        self.project = nn.Linear(enc_width, latent_dim * dist_nparams)

        self.freeze_mask = False  # set to True to reuse the same mask multiple times


    @property
    def freeze_mask(self):
        '''When True, the previously computed mask will be used on new inputs, instead of creating a new one.'''
        return self._freeze_mask

    @freeze_mask.setter
    def freeze_mask(self, val: bool):
        self._freeze_mask = val

    @staticmethod
    def pos_encoding(n: int, d: int, k: int=10000):
        '''Create sine-cosine positional embeddings.

        Args:
            n: the number of embedding vectors, corresponding to the number of tokens (patches) in the image.
            d: the dimension of the embeddings
            k: value that determines the maximum frequency (10,000 by default)
        
        Returns:
            (n, d) tensor of position encoding vectors
        '''
        x = torch.meshgrid(
            torch.arange(n, dtype=torch.float32),
            torch.arange(d, dtype=torch.float32),
            indexing='ij'
        )
        pos = torch.zeros_like(x[0])
        pos[:, ::2] = x[0][:, ::2].div(torch.pow(k, x[1][:, ::2].div(d // 2))).sin_()
        pos[:, 1::2] = x[0][:,1::2].div(torch.pow(k, x[1][:,1::2].div(d // 2))).cos_()
        return pos

    @staticmethod
    def generate_mask_index(bs: int, n_tok: int, device: str='cpu'):
        '''Create a randomly permuted token-index tensor for determining which tokens to mask.

        Args:
            bs: batch size
            n_tok: number of tokens per image
            device: the device where the tensors should be created

        Returns:
            (bs, 1) tensor of batch indices [0, 1, ..., bs - 1]^T
            (bs, n_tok) tensor of token indices, randomly permuted
        '''
        idx = torch.rand(bs, n_tok, device=device).argsort(dim=1)
        return idx

    @staticmethod
    def select_tokens(x: torch.Tensor, idx: torch.Tensor):
        '''Return the tokens from `x` corresponding to the indices in `idx`.
        '''
        idx = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return x.gather(dim=1, index=idx)

    def image_as_tokens(self, x: torch.Tensor):
        '''Reshape an image of shape (b, c, h, w) to a set of vectorized patches
        of shape (b, h*w/p^2, c*p^2). In other words, the set of non-overlapping
        patches of size (3, p, p) in the image are turned into vectors (tokens); 
        dimension 1 of the output indexes each patch.
        '''
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(b, (h * w) // p**2, c * p * p)
        return x

    def tokens_as_image(self, x: torch.Tensor):
        '''Reshape a set of token vectors into an image. This is the reverse operation
        of `image_as_tokens`.
        '''
        b = x.shape[0]
        im, p = self.image_size, self.patch_size
        hh, ww = im[0] // p, im[1] // p
        x = x.reshape(b, hh, ww, 3, p, p).permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(b, 3, p * hh, p * ww)
        return x

    def masked_image(self, x: torch.Tensor):
        '''Return a copy of the image batch, with the masked patches set to 0. Used
        for visualization.
        '''
        x = self.image_as_tokens(x).clone()
        bidx = torch.arange(x.shape[0], device=x.device)[:, None]
        x[bidx, self.idx[:, int(self.keep * self.n):]] = 0
        return self.tokens_as_image(x)

    def embed(self, x: torch.Tensor):
        return self.embed_conv(x).flatten(2).transpose(1, 2)

    def mask_input(self, x: torch.Tensor):
        '''Mask the image patches uniformly at random, as described in the paper: the patch tokens are
        randomly permuted (per image), and the first N are returned, where N corresponds to percentage
        of patches kept (not masked).
        
        Returns the masked (truncated) tokens. The mask indices are saved as `self.bidx` and `self.idx`.
        '''
        # create a new mask if self.freeze_mask is False, or if no mask has been created yet
        if not hasattr(self, 'idx') or not self.freeze_mask:
            self.idx = self.generate_mask_index(x.shape[0], x.shape[1], x.device)

        k = int(self.keep * self.n)
        x = self.select_tokens(x, self.idx[:, :k])
        return x

    def forward_features(self, x: torch.Tensor):
        shuffled_idx = torch.randperm(x.shape[0])
        shuffled_x = x[shuffled_idx]
        x = self.embed(torch.cat([shuffled_x,x], dim=-2))
        x = x + self.pos_encoder
        b,n,d = x.shape
        view_0 = x[:, :n//2]
        view_1 = x[:, n//2:]
        view_1 = self.mask_input(view_1)
        x = torch.cat([view_0, view_1], dim=1)
        x = self.encoder(x)

        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        x = self.project(torch.relu(x))

        return {'stats_qzx': x.mean(1).view(-1, self.latent_dim, self.dist_nparams)}
        

if __name__ == '__main__':
    model = Encoder(
        image_size=(3, 64, 64),
        latent_dim=10
    )
    fakein = torch.randn(64, 3, 64, 64)
    out = model(fakein)
    print(out['stats_qzx'].shape)