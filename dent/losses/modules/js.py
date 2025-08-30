import torch
import matplotlib.pyplot as plt

def js_divergence_torch(mu1, logvar1, mu2, logvar2):
    """
    Calculate the Jensen-Shannon divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu1, mu2 (torch.Tensor): Means of the Gaussian distributions [batch_size, dim]
    logvar1, logvar2 (torch.Tensor): Log variances of the Gaussian distributions [batch_size, dim]
    
    Returns:
    torch.Tensor: The Jensen-Shannon divergence between the distributions [batch_size]
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

def test_js_divergence_bounds():
    """
    Test the bounds of JS divergence with various distribution parameters
    """
    torch.manual_seed(42)
    
    js_values = []
    mean_diffs = []
    
    dim = 2
    n_tests = 1000
    
    for _ in range(n_tests):
        # Generate random parameters
        mu1 = torch.randn(1, dim)
        mu2 = torch.randn(1, dim)
        logvar1 = torch.randn(1, dim)
        logvar2 = torch.randn(1, dim)
        
        js_div = js_divergence_torch(mu1, logvar1, mu2, logvar2)
        js_values.append(js_div.item())
        
        mean_diff = torch.norm(mu1 - mu2).item()
        mean_diffs.append(mean_diff)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(js_values, bins=50)
    plt.title('Distribution of JS Divergence Values')
    plt.xlabel('JS Divergence')
    plt.ylabel('Count')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=1, color='r', linestyle='--', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.scatter(mean_diffs, js_values, alpha=0.5)
    plt.title('JS Divergence vs Mean Difference')
    plt.xlabel('Mean Difference (L2 norm)')
    plt.ylabel('JS Divergence')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Min JS divergence: {min(js_values):.6f}")
    print(f"Max JS divergence: {max(js_values):.6f}")
    print(f"Mean JS divergence: {sum(js_values)/len(js_values):.6f}")

def test_special_cases():
    """
    Test JS divergence for special cases
    """
    print("\nSpecial cases:")
    
    # Case 1: Identical distributions
    mu = torch.zeros(1, 2)
    logvar = torch.zeros(1, 2)
    js_same = js_divergence_torch(mu, logvar, mu, logvar)
    print(f"JS divergence for identical distributions: {js_same.item():.6f}")
    
    # Case 2: Very different means
    mu1 = torch.zeros(1, 2)
    mu2 = torch.ones(1, 2) * 10
    logvar = torch.zeros(1, 2)
    js_diff = js_divergence_torch(mu1, logvar, mu2, logvar)
    print(f"JS divergence for distant distributions: {js_diff.item():.6f}")
    
    # Case 3: Very different variances
    mu = torch.zeros(1, 2)
    logvar1 = torch.zeros(1, 2)
    logvar2 = torch.ones(1, 2) * 4
    js_var = js_divergence_torch(mu, logvar1, mu, logvar2)
    print(f"JS divergence for different variances: {js_var.item():.6f}")

# Run tests
test_js_divergence_bounds()
test_special_cases()