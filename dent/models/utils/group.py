import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class LieGroupProcessor(nn.Module):
    def __init__(self, d, subgroup_sizes_ls, subspace_sizes_ls, lie_alg_init_scale=0.1, no_exp=False,repara=False):
        super(LieGroupProcessor, self).__init__()
        self.d = d
        self.subgroup_sizes_ls = subgroup_sizes_ls
        self.subspace_sizes_ls = subspace_sizes_ls
        self.lie_alg_init_scale = lie_alg_init_scale
        self.no_exp = no_exp
        self.repara = repara
        
        # Initialize Lie algebra basis for each latent dimension.
        self.lie_alg_basis_ls = nn.ParameterList([])
        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            mat_dim = int(math.sqrt(subgroup_size_i))
            assert mat_dim * mat_dim == subgroup_size_i
            for j in range(self.subspace_sizes_ls[i]):
                lie_alg_tmp, var_tmp = self.init_alg_basis(i, j, mat_dim, lie_alg_init_scale)
                self.lie_alg_basis_ls.append(lie_alg_tmp)

        if self.no_exp:
            in_size = sum(self.subspace_sizes_ls)
            out_size = sum(self.subgroup_sizes_ls)
            self.fake_exp = nn.Sequential(
                    nn.Linear(in_size, in_size * 4),
                    nn.ReLU(True),
                    nn.Linear(in_size * 4, out_size))
            for p in self.fake_exp.modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                        isinstance(p, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(p.weight)

        if self.repara:
            self.to_means = nn.ModuleList([])
            self.to_logvar = nn.ModuleList([])
            for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
                self.to_means.append(
                    nn.Sequential(
                        nn.Linear(subgroup_size_i, subgroup_size_i * 4),
                        nn.ReLU(True),
                        nn.Linear(subgroup_size_i * 4, subspace_sizes_ls[i]),
                    ))
                self.to_logvar.append(
                    nn.Sequential(
                        nn.Linear(subgroup_size_i, subgroup_size_i * 4),
                        nn.ReLU(True),
                        nn.Linear(subgroup_size_i * 4, subspace_sizes_ls[i]),
                    ))
            for p in self.to_means.modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                    torch.nn.init.xavier_uniform_(p.weight)
            for p in self.to_logvar.modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                    torch.nn.init.xavier_uniform_(p.weight)
    
    def val_exp(self, x, lie_alg_basis_ls):
        lie_alg_basis_ls = [p * 1. for p in lie_alg_basis_ls
                            ]  # For torch.cat, convert param to tensor.
        lie_alg_basis = torch.cat(lie_alg_basis_ls,
                                  dim=0)[np.newaxis,
                                         ...]  # [1, lat_dim, mat_dim, mat_dim]
        lie_alg_mul = x[
            ..., np.newaxis, np.
            newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
        lie_alg = torch.sum(lie_alg_mul, dim=1)  # [b, mat_dim, mat_dim]
        lie_group = torch.matrix_exp(lie_alg)  # [b, mat_dim, mat_dim]
        return lie_group

    def train_exp(self, x, lie_alg_basis_ls, mat_dim):
        lie_alg_basis_ls = [p * 1. for p in lie_alg_basis_ls
                            ]  # For torch.cat, convert param to tensor.
        lie_alg_basis = torch.cat(lie_alg_basis_ls,
                                  dim=0)[np.newaxis,
                                         ...]  # [1, lat_dim, mat_dim, mat_dim]
        lie_group = torch.eye(mat_dim, dtype=x.dtype).to(
            x.device)[np.newaxis, ...]  # [1, mat_dim, mat_dim]
        lie_alg = 0.
        latents_in_cut_ls = [x]
        for masked_latent in latents_in_cut_ls:
            lie_alg_sum_tmp = torch.sum(
                masked_latent[..., np.newaxis, np.newaxis] * lie_alg_basis,
                dim=1)
            lie_alg += lie_alg_sum_tmp  # [b, mat_dim, mat_dim]
            lie_group_tmp = torch.matrix_exp(lie_alg_sum_tmp)
            lie_group = torch.matmul(lie_group,
                                     lie_group_tmp)  # [b, mat_dim, mat_dim]
        return lie_group

    def init_alg_basis(self, i, j, mat_dim, lie_alg_init_scale):
        lie_alg_tmp = nn.Parameter(
            torch.normal(mean=torch.zeros(1, mat_dim, mat_dim), std=lie_alg_init_scale),
            requires_grad=True
            )
        var_tmp = nn.Parameter(torch.normal(torch.zeros(1, 1), lie_alg_init_scale))
        
        torch.nn.init.xavier_uniform_(lie_alg_tmp)
        torch.nn.init.xavier_uniform_(var_tmp)
        
        return lie_alg_tmp, var_tmp
    
    def calc_basis_mul_ij(self, lie_alg_basis_ls_param):
        lie_alg_basis_ls = [alg_tmp * 1. for alg_tmp in lie_alg_basis_ls_param]
        lie_alg_basis = torch.cat(lie_alg_basis_ls,
                                  dim=0)[np.newaxis,
                                         ...]  # [1, lat_dim, mat_dim, mat_dim]
        _, lat_dim, mat_dim, _ = list(lie_alg_basis.size())
        lie_alg_basis_col = lie_alg_basis.view(lat_dim, 1, mat_dim, mat_dim)
        lie_alg_basis_outer_mul = torch.matmul(
            lie_alg_basis,
            lie_alg_basis_col)  # [lat_dim, lat_dim, mat_dim, mat_dim]
        hessian_mask = 1. - torch.eye(
            lat_dim, dtype=lie_alg_basis_outer_mul.dtype
        )[:, :, np.newaxis, np.newaxis].to(lie_alg_basis_outer_mul.device)
        lie_alg_basis_mul_ij = lie_alg_basis_outer_mul * hessian_mask  # XY
        return lie_alg_basis_mul_ij
    
    def calc_hessian_loss(self, lie_alg_basis_mul_ij, i):
        hessian_loss = torch.mean(
            torch.sum(torch.square(lie_alg_basis_mul_ij), dim=[2, 3]))
        return hessian_loss
    
    def calc_commute_loss(self, lie_alg_basis_mul_ij, i):
        lie_alg_commutator = lie_alg_basis_mul_ij - lie_alg_basis_mul_ij.permute(
            1, 0, 2, 3)
        commute_loss = torch.mean(
            torch.sum(torch.square(lie_alg_commutator), dim=[2, 3]))
        return commute_loss
    
    def group_loss(self):
        b_idx = 0
        hessian_loss = 0.
        commute_loss = 0.
        for i, subspace_size in enumerate(self.subspace_sizes_ls):
            e_idx = b_idx + subspace_size
            if subspace_size > 1:
                mat_dim = int(math.sqrt(self.subgroup_sizes_ls[i]))
                assert list(self.lie_alg_basis_ls[b_idx].size())[-1] == mat_dim
                lie_alg_basis_mul_ij = self.calc_basis_mul_ij(
                    self.lie_alg_basis_ls[b_idx:e_idx])  # XY
                hessian_loss += self.calc_hessian_loss(lie_alg_basis_mul_ij, i)
                commute_loss += self.calc_commute_loss(lie_alg_basis_mul_ij, i)
            b_idx = e_idx
        
        # if self.repara:
        #     rec_loss = torch.mean(
        #         torch.sum(torch.square(group_feats_E - group_feats_G), dim=1))
        # else:
        #     rec_loss = 0.
        
        return hessian_loss, commute_loss 
    
    def unwrap(self, x):
        return torch.split(x, x.shape[1]//2, dim=1)

    def reparametrise(self, mu, lv):
        if self.training:
            std = torch.exp(0.5 * lv)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu
    
    def forward(self, z):
        # Assume z is of shape (batch, d) and is the input latent vector.

        if self.repara:
            means_ls, logvars_ls = [], []
            b_idx = 0
            for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
                e_idx = b_idx + subgroup_size_i
                means_ls.append(self.to_means[i](z[:, b_idx:e_idx]))
                logvars_ls.append(self.to_logvar[i](z[:, b_idx:e_idx]))
                b_idx = e_idx
            outs = torch.cat(means_ls + logvars_ls, dim=-1)
            mu, lv = self.unwrap(outs)
            z = self.reparametrise(mu, lv)
        
        batch_size, latent_dim = z.size()

        if self.no_exp:
            lie_group_tensor = self.fake_exp(z)
        else:
            assert latent_dim == sum(self.subspace_sizes_ls), f'{latent_dim} != {sum(self.subspace_sizes_ls)}'

            # Calc exp.
            lie_group_tensor_ls = []
            b_idx = 0
            for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
                mat_dim = int(math.sqrt(subgroup_size_i))
                e_idx = b_idx + self.subspace_sizes_ls[i]
                if self.subspace_sizes_ls[i] > 1:
                    if not self.training:
                        lie_subgroup = self.val_exp(
                            z[:, b_idx:e_idx],
                            self.lie_alg_basis_ls[b_idx:e_idx])
                    else:
                        lie_subgroup = self.train_exp(
                            z[:, b_idx:e_idx],
                            self.lie_alg_basis_ls[b_idx:e_idx], mat_dim)
                else:
                    lie_subgroup = self.val_exp(z[:, b_idx:e_idx],
                                                self.lie_alg_basis_ls[b_idx:e_idx])
                lie_subgroup_tensor = lie_subgroup#.view(-1, mat_dim * mat_dim)
                lie_group_tensor_ls.append(lie_subgroup_tensor)
                b_idx = e_idx
            
            lie_group_tensor = torch.cat(lie_group_tensor_ls,
                                         dim=1)  # [b, group_feat_size]
        # print(lie_group_tensor.shape)
        return lie_group_tensor


if __name__=='__main__':

    # Example usage:
    # Define the dimensions and subgroup sizes.
    b = 4
    subgroup_sizes_ls = [100,100]  # Example subgroup sizes.
    subspace_sizes_ls = [256,256]  # Corresponding dimensions for each subgroup.
    repara = True  # Whether to use reparameterization.

    # Initialize the LieGroupProcessor.
    lie_group_processor = LieGroupProcessor(sum(subspace_sizes_ls), subgroup_sizes_ls, subspace_sizes_ls, repara=repara)

    # Create a random latent vector.
    z = torch.randn(b, sum(subgroup_sizes_ls)) if repara else torch.randn(b, sum(subspace_sizes_ls))
    lie_group_feature = lie_group_processor(z)
    group_loss = lie_group_processor.group_loss(z, lie_group_feature) if repara else None

    print(lie_group_feature.shape, group_loss)