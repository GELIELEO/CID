import torch

def _matrix_poly(matrix, d):
    # 获取矩阵的维度
    batch_size, n, _ = matrix.shape
    # 创建单位矩阵并扩展到批处理大小
    eye = torch.eye(n, device=matrix.device).unsqueeze(0).repeat(batch_size, 1, 1)
    # 计算多项式
    x = eye + torch.div(matrix, d)
    return torch.linalg.matrix_power(x, d)

def batch_dagness(A, m):
    if A.dim() != 3 or A.shape[1] != A.shape[2]:
        raise ValueError("Input A must be a 3D tensor with shape (batch_size, n, n).")
    # 计算每个矩阵的平方
    A_squared = torch.bmm(A, A)
    # 计算多项式
    expm_A = _matrix_poly(A_squared, m)
    # 计算迹
    h_A = torch.sum(torch.diagonal(expm_A, dim1=1, dim2=2), dim=1) - m
    return h_A

if __name__=='__main__':
    # 示例使用
    batch_size = 4
    n = 10
    m = 10
    A = torch.randn(batch_size, n, n)
    A = torch.ones(batch_size, n, n)
    # A = torch.zeros(batch_size, n, n)
    # A[:, 0, 0] = 1
    # A[:, 1, 1] = 1
    # A[:, 2, 1] = 1
    # A[:, 1, 2] = 1
    result = batch_dagness(A, m)
    print("D'Agness for each matrix in the batch:", result)