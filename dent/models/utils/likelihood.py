import torch

def log_likelihood_normal(samples, mean, std):
    """
    使用正态分布公式计算对数似然。

    参数:
    samples (torch.Tensor): 样本数据，形状为 (n_samples,)
    mean (torch.Tensor): 正态分布的均值
    std (torch.Tensor): 正态分布的标准差，必须大于0

    返回:
    torch.Tensor: 样本的对数似然值，形状为 (n_samples,)
    """
    # 确保mean和std也是张量
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        mean = torch.as_tensor(mean, dtype=samples.dtype, device=samples.device)
        std = torch.as_tensor(std, dtype=samples.dtype, device=samples.device)
    # 计算对数似然的常数项
    log_const = -0.5 * torch.log(2 * torch.pi * std**2)
    
    # 计算对数似然的变量项
    variable_term = -0.5 * ((samples - mean) / std) ** 2
    
    # 将常数项和变量项相加得到对数似然
    log_likelihoods = log_const + variable_term
    
    return log_likelihoods


if __name__ == "__main__":
    # 示例使用
    # 假设我们有以下样本、均值和标准差
    normal = torch.distributions.Normal(loc=10, scale=1e-0)
    samples = normal.sample(sample_shape=(3,3))
    mean = 10.0
    std = 1e-0

    # 计算对数似然
    log_likelihood_values = log_likelihood_normal(samples, mean, std)
    print(log_likelihood_values)
    print(normal.log_prob(samples))