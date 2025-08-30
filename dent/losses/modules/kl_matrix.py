import torch

def compute_kl_divergence_matrix(batch):
    """
    计算批次中样本两两之间的 KL 散度矩阵。
    
    参数:
    batch: torch.Tensor, 形状为 (1024, 10, 2)
           最后一个维度中，[:, :, 0] 是均值，[:, :, 1] 是 logvar
    
    返回:
    kl_matrix: torch.Tensor, 形状为 (1024, 1024)
               表示样本两两之间的 KL 散度
    """
    n_samples, n_dim = batch.shape[:2]
    
    # 提取均值和方差
    means = batch[:, :, 0]  # (1024, 10)
    logvars = batch[:, :, 1]  # (1024, 10)
    vars = torch.exp(logvars)  # (1024, 10)
    
    # 计算均值差的平方
    mean_diff = means.unsqueeze(1) - means.unsqueeze(0)  # (1024, 1024, 10)
    mean_diff_sq = mean_diff.pow(2)  # (1024, 1024, 10)
    
    # 计算方差的比值和差
    var_ratio = vars.unsqueeze(1) / vars.unsqueeze(0)  # (1024, 1024, 10)
    logvar_diff = logvars.unsqueeze(1) - logvars.unsqueeze(0)  # (1024, 1024, 10)
    
    # 计算 KL 散度
    kl_div = 0.5 * (
        var_ratio + mean_diff_sq / vars.unsqueeze(0) - 1 - logvar_diff
    ).sum(dim=2)  # (1024, 1024)
    
    return kl_div

# 测试函数
if __name__ == "__main__":
    # 创建一个随机批次进行测试
    batch = torch.rand(10, 10, 2)
    # batch[:, :, 1] = torch.exp(batch[:, :, 1])  # 确保 logvar 是正数
    # batch[0] = 2
    
    kl_matrix = compute_kl_divergence_matrix(batch)
    print("KL 散度矩阵的形状:", kl_matrix.shape)
    print("KL 散度矩阵的一些值:\n", kl_matrix)
    print(kl_matrix.argmax(dim=1))


    from scipy.optimize import linear_sum_assignment
    row, col = linear_sum_assignment(kl_matrix.numpy())
    print(row, col)