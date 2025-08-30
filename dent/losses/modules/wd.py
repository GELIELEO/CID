import torch

def wasserstein_gaussian(mu1, logvar1, mu2, logvar2):
    """
    Calculate the 2-Wasserstein distance between two multivariate Gaussian distributions.
    
    For Gaussian distributions P1 = N(mu1, Sigma1) and P2 = N(mu2, Sigma2), 
    W_2^2(P1, P2) = ||mu1 - mu2||^2 + Tr(Sigma1 + Sigma2 - 2(Sigma1^(1/2)Sigma2Sigma1^(1/2))^(1/2))
    
    For diagonal covariance matrices, this simplifies to:
    W_2^2(P1, P2) = ||mu1 - mu2||^2 + ||sigma1 - sigma2||^2
    
    Parameters:
    mu1, mu2 (torch.Tensor): Means of shape [batch_size, dim]
    logvar1, logvar2 (torch.Tensor): Log variances of shape [batch_size, dim]
    
    Returns:
    torch.Tensor: Wasserstein distance of shape [batch_size]
    """
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

def test_wasserstein_distance():
    """
    Test the Wasserstein distance with various cases
    """
    # Test case 1: Identical distributions
    print("\nTest Case 1: Identical distributions")
    mu = torch.zeros(1, 2)
    logvar = torch.zeros(1, 2)
    w_same = wasserstein_gaussian(mu, logvar, mu, logvar)
    print(f"W distance for identical distributions: {w_same.item():.6f}")
    
    # Test case 2: Different means, same variance
    print("\nTest Case 2: Different means, same variance")
    mu1 = torch.zeros(1, 2)
    mu2 = torch.ones(1, 2)
    logvar = torch.zeros(1, 2)
    w_diff_mean = wasserstein_gaussian(mu1, logvar, mu2, logvar)
    print(f"W distance for different means: {w_diff_mean.item():.6f}")
    
    # Test case 3: Same mean, different variances
    print("\nTest Case 3: Same mean, different variances")
    mu = torch.zeros(1, 2)
    logvar1 = torch.zeros(1, 2)
    logvar2 = torch.ones(1, 2)
    w_diff_var = wasserstein_gaussian(mu, logvar1, mu, logvar2)
    print(f"W distance for different variances: {w_diff_var.item():.6f}")
    
    # Test case 4: Different means and variances
    print("\nTest Case 4: Different means and variances")
    mu1 = torch.zeros(1, 2)
    mu2 = torch.ones(1, 2)
    logvar1 = torch.zeros(1, 2)
    logvar2 = torch.ones(1, 2)
    w_diff_both = wasserstein_gaussian(mu1, logvar1, mu2, logvar2)
    print(f"W distance for different means and variances: {w_diff_both.item():.6f}")
    
    # Test case 5: Batch processing
    print("\nTest Case 5: Batch processing")
    batch_size = 3
    mu1 = torch.randn(batch_size, 2)
    mu2 = torch.randn(batch_size, 2)
    logvar1 = torch.randn(batch_size, 2)
    logvar2 = torch.randn(batch_size, 2)
    w_batch = wasserstein_gaussian(mu1, logvar1, mu2, logvar2)
    print(f"W distances for batch: {w_batch}")

# Run tests
test_wasserstein_distance()

# Visualization of W distance properties
def visualize_w_distance():
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate test points
    n_points = 100
    mean_diffs = np.linspace(0, 5, n_points)
    w_distances = []
    
    mu1 = torch.zeros(1, 1)
    logvar1 = torch.zeros(1, 1)
    logvar2 = torch.zeros(1, 1)
    
    for diff in mean_diffs:
        mu2 = torch.tensor([[diff]], dtype=torch.float32)
        w_dist = wasserstein_gaussian(mu1, logvar1, mu2, logvar2)
        w_distances.append(w_dist.item())
    
    plt.figure(figsize=(10, 5))
    
    # Plot W distance vs mean difference
    plt.subplot(1, 2, 1)
    plt.plot(mean_diffs, w_distances)
    plt.title('W Distance vs Mean Difference')
    plt.xlabel('Mean Difference')
    plt.ylabel('W Distance')
    
    # Plot W distance vs variance difference
    var_diffs = np.linspace(0, 2, n_points)
    w_distances_var = []
    mu = torch.zeros(1, 1)
    logvar1 = torch.zeros(1, 1)
    
    for diff in var_diffs:
        logvar2 = torch.tensor([[np.log(1 + diff)]], dtype=torch.float32)
        w_dist = wasserstein_gaussian(mu, logvar1, mu, logvar2)
        w_distances_var.append(w_dist.item())
    
    plt.subplot(1, 2, 2)
    plt.plot(var_diffs, w_distances_var)
    plt.title('W Distance vs Variance Difference')
    plt.xlabel('Variance Difference')
    plt.ylabel('W Distance')
    
    plt.tight_layout()
    plt.show()

# Run visualization
visualize_w_distance()

