import torch

def kl_divergence_two_gaussian(mu1, logvar1, mu2, logvar2):
    """
    计算两个多元高斯分布之间的KL散度。
    
    参数:
    - mu1: 第一个batch的均值张量，形状为 (batch_size, num_features)
    - logvar1: 第一个batch的log方差张量，形状为 (batch_size, num_features)
    - mu2: 第二个batch的均值张量，形状为 (batch_size, num_features)
    - logvar2: 第二个batch的log方差张量，形状为 (batch_size, num_features)
    
    返回:
    - kl_div: KL散度的张量，形状为 (batch_size,)
    """
    # 计算方差的指数
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    
    # 计算KL散度
    kl_div = 0.5 * (logvar2 - logvar1 + (var1 + (mu1 - mu2) ** 2) / var2 - 1)
    kl_div = torch.sum(kl_div, dim=1)  # 沿特征维度求和
    return kl_div


def js_divergence_two_gaussian(mu1, logvar1, mu2, logvar2):
    """
    计算两个多元高斯分布之间的JS散度。
    
    参数:
    - mu1: 第一个batch的均值张量，形状为 (batch_size, num_features)
    - logvar1: 第一个batch的log方差张量，形状为 (batch_size, num_features)
    - mu2: 第二个batch的均值张量，形状为 (batch_size, num_features)
    - logvar2: 第二个batch的log方差张量，形状为 (batch_size, num_features)
    
    返回:
    - js_div: JS散度的张量，形状为 (batch_size,)
    """
    # Convert log variance to variance
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    
    # Calculate the parameters of the average distribution M
    mu_m = (mu1 + mu2) / 2
    var_m = (var1 + var2) / 2
    logvar_m = torch.log(var_m)
    
    # Calculate KL(P||M)
    kl_pm = 0.5 * (
        torch.sum(logvar_m - logvar1, dim=-1) + 
        torch.sum((var1 + (mu1 - mu_m).pow(2)) / var_m, dim=-1) - 
        mu1.size(-1)
    )
    
    # Calculate KL(Q||M)
    kl_qm = 0.5 * (
        torch.sum(logvar_m - logvar2, dim=-1) + 
        torch.sum((var2 + (mu2 - mu_m).pow(2)) / var_m, dim=-1) - 
        mu2.size(-1)
    )
    
    # Calculate JS divergence and normalize
    js_div = 0.5 * (kl_pm + kl_qm) / torch.log(torch.tensor(2.0))


    return js_div



def wasserstein_distance_two_gaussian(mu1, logvar1, mu2, logvar2):

    # Convert log variance to standard deviation
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    std1 = torch.sqrt(var1)
    std2 = torch.sqrt(var2)
    
    # Mean term: ||mu1 - mu2||^2
    mean_term = torch.sum((mu1 - mu2)**2, dim=-1)
    
    # Variance term: ||sigma1 - sigma2||^2
    var_term = torch.sum((std1 - std2)**2, dim=-1)
    
    # Total Wasserstein distance
    w_dist = torch.sqrt(mean_term + var_term)
    
    return w_dist

if __name__ == "__main__":

    # 示例使用
    # 假设我们有两个batch，每个batch有3个样本，每个样本是2维的分布
    batch_size = 3
    latent_dim = 1
    mu1 = torch.zeros(batch_size, latent_dim)
    logvar1 = torch.ones(batch_size, latent_dim)
    mu2 = torch.zeros(batch_size, latent_dim)
    logvar2 = torch.ones(batch_size, latent_dim)

    # 计算KL散度
    kl_div = kl_divergence_two_gaussian(mu1, logvar1, mu2, logvar2)
    print(kl_div)
    kl_div = kl_divergence_two_gaussian(mu2, logvar2, mu1, logvar1)
    print(kl_div)