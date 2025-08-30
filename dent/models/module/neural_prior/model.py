## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import normflows as nf



class SpikeVariationalDequantization(nf.flows.Flow):

    def __init__(self, var_flows, quantize_enabled=True):
        """
        Inputs:
            var_flows - A list of flow transformations to use for modeling q(u|x)
            alpha - Small constant, see Dequantization for details
        """
        super().__init__()
        self.flows = nn.ModuleList(var_flows)
        # self.sigmoid = zuko.transforms.SigmoidTransform()
        self.quants = 100
        self.alpha=1e-5
        self.quantize_enabled = quantize_enabled

    def sigmoid(self, z, ldj, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            ldj += (-z-2*F.softplus(-z)).sum(dim=-1)
            z = torch.sigmoid(z)
            # Reversing scaling for numerical stability
            ldj -= torch.log(torch.tensor(1-self.alpha)) * torch.prod(torch.tensor(z.shape[1:]))
            z = (z - 0.5 * self.alpha) / (1-self.alpha)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            ldj += torch.log(torch.tensor(1-self.alpha)) * torch.prod(torch.tensor(z.shape[1:]))
            ldj += (-torch.log(z) - torch.log(1-z)).sum(dim=-1)
            z = torch.log(z) - torch.log(1-z)
        return z, ldj

    def dequant(self, z, ldj):

        ctx = (z / self.quants) * 2 - 1 # We condition the flows on x
        deq_noise_0 = torch.rand_like(z).detach()

        # Prior of u is a uniform distribution as before
        # As most flow transformations are defined on [-infinity,+infinity], we apply an inverse sigmoid first.
        # deq_noise = self.sigmoid.inv(deq_noise_0)
        # ldj += self.sigmoid.log_abs_det_jacobian(deq_noise_0, deq_noise).sum(-1)
        deq_noise, ldj = self.sigmoid(deq_noise_0, ldj, reverse=True)

        for flow in self.flows[::-1]:
            deq_noise, _ldj = flow.inverse(deq_noise, context=ctx)
            ldj += _ldj

        # deq_noise_1 = self.sigmoid(deq_noise)
        # ldj -= self.sigmoid.log_abs_det_jacobian(deq_noise, deq_noise_1).sum(-1)
        deq_noise_1, ldj = self.sigmoid(deq_noise, ldj, reverse=False)

        z = (z + deq_noise_1) / self.quants
        ldj -= torch.log(torch.tensor(self.quants)) * torch.prod(torch.tensor(z.shape[1:]))

        return z, ldj

    def inverse(self, z):
        ldj = torch.zeros(z.shape[0], device=z.device)
        if self.quantize_enabled:
            z, ldj = self.dequant(z, ldj)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        return z, ldj

    def forward(self, z):
        ldj = torch.zeros(z.shape[0], device=z.device)
        # z = self.sigmoid(z_0)
        # ldj -= self.sigmoid.log_abs_det_jacobian(z_0, z).sum(-1)
        # z = z * self.quants
        # ldj -= torch.log(torch.tensor(self.quants)) * torch.prod(torch.tensor(z.shape[1:]))
        # z = torch.floor(z).clamp(min=0, max=self.quants-1)
        if self.quantize_enabled:
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            z = z * self.quants
            ldj += torch.log(torch.tensor(self.quants)) * torch.prod(torch.tensor(z.shape[1:]))
            z = torch.floor(z).clamp(min=0, max=self.quants-1)
        return z, ldj
    
class SpikeModel(nn.Module):
    def __init__(self, num_units):
        super().__init__()

        latent_size = num_units
        context_size = latent_size
        hidden_units = 256
        hidden_layers = 2
        flows, vdq_flow = [], []

        '''
        top -> bottom (forward)
        base -> target
        '''
        for i in range(16):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
            flows += [nf.flows.LULinearPermute(latent_size)]
            # net = nf.nets.LipschitzMLP([latent_size] + [hidden_units] * (hidden_layers - 1) + [latent_size], init_zeros=True, lipschitz_const=0.9)
            # flows += [nf.flows.Residual(net, reduce_memory=True)]
            # flows += [nf.flows.ActNorm(latent_size)]

        for i in range(3):
            vdq_flow += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, num_context_channels=context_size)]
            vdq_flow += [nf.flows.LULinearPermute(latent_size)]
        flows += [SpikeVariationalDequantization(vdq_flow)]

        q0 = nf.distributions.DiagGaussian(latent_size, trainable=True)
        # q0 = nf.distributions.GaussianMixture(latent_size, latent_size, loc=torch.randn(latent_size, latent_size), scale=torch.ones(latent_size, latent_size), trainable=True)

        self.flow = nf.NormalizingFlow(q0, flows)
    
    def forward(self, x):
        return self.flow(x)
    
    def log_prob(self, x):
        return self.flow.log_prob(x)
    
    def forward_kld(self, x):
        return self.flow.forward_kld(x)
    
    def reverse_kld(self, n):
        return self.flow.reverse_kld(n)
    
    def sample(self, n=1):
        return self.flow.sample(n)
    
    def inverse(self, x):
        return self.flow.inverse(x)
    
    def inverse_and_log_det(self, x):
        return self.flow.inverse_and_log_det(x)
    
    def forward_and_log_det(self, x):
        return self.flow.forward_and_log_det(x)




if __name__ == "__main__":
    latent_size = 2
    context_size = latent_size
    hidden_units = 256
    hidden_layers = 2
    vdq_flow = []
    for i in range(3):
        vdq_flow += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, num_context_channels=context_size)]
        vdq_flow += [nf.flows.LULinearPermute(latent_size)]
    model = SpikeVariationalDequantization(vdq_flow)
    
    fake_input = torch.rand(1, latent_size)
    fake_context = torch.randn(1, context_size)
    ldj_0 = torch.zeros(fake_context.shape[0])
    out_f, ldj_f = model.sigmoid(fake_input, ldj_0)
    ldj_0 = torch.zeros(fake_context.shape[0])
    out_r, ldj_r = model.sigmoid(out_f, ldj_0, reverse=True)
    print(fake_input, out_f, out_r, ldj_f, ldj_r)

    fake_input = torch.ones(1, latent_size)*45
    ldj_0 = torch.zeros(fake_context.shape[0])
    out_f, ldj_f = model.inverse(fake_input)
    ldj_0 = torch.zeros(fake_context.shape[0])
    out_r, ldj_r = model.forward(out_f)
    print(fake_input, out_f, out_r, ldj_f, ldj_r)
    
    
    model = SpikeModel(latent_size)
    fake_input = torch.ones(1, latent_size)*3
    ldj_0 = torch.zeros(fake_context.shape[0])
    out_f, ldj_f = model.inverse_and_log_det(fake_input)
    ldj_0 = torch.zeros(fake_context.shape[0])
    out_r, ldj_r = model.forward_and_log_det(out_f)
    print(fake_input, out_f, out_r, ldj_f, ldj_r)
    sample = model.sample(1)
    log_p = model.log_prob(sample[0])
    print(sample, log_p)