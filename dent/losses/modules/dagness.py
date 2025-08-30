import torch


def _matrix_poly(matrix, d):
    x = torch.eye(d).to(matrix.device)+ torch.div(matrix, d)
    return torch.matrix_power(x, d)

def dagness(A, m):
    expm_A = _matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

if __name__ == '__main__':
    A = torch.randn(3,3)
    A = torch.eye(3)
    A = torch.ones(64,3,3)
    # A = torch.zeros(3,3)
    # A[0,1] = 1
    # A[1,0] = 1
    # A[1,2] = 1
    # A[2,1] = 1
    # A[1,1] = 1
    out = dagness(A, 3)
    print(out)