import gpytorch
import torch

# GPytorch Models for exact GP
class ExactGPModel(gpytorch.models.ExactGP):
    """
    Standard Exact GP model using a Matern kernel (nu=2.5).
    Uses KeOps for efficient computation on large datasets.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        # ScaleKernel allows the model to learn the variance (outputscale)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.MaternKernel(nu=2.5)) 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class TransformedGPModel(gpytorch.models.ExactGP):
    """
    GP model that applies a fixed transformation to the input features 
    before computing the covariance. This represents the 'Ideal' GP 
    where the spatial distortion is known.
    """
    def __init__(self, train_x, train_y, likelihood, function, lengthscale):
        super(TransformedGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.MaternKernel(nu=2.5))
        self.function = function
        self.lengthscale = lengthscale

    def forward(self, x):
        device = x.device
        # Transform the inputs using the provided function
        # Note: Input is moved to CPU to handle NumPy-based transformations
        x_np = self.function(x.cpu().numpy(), scale=self.lengthscale)
        x_transformed = torch.tensor(x_np, dtype=torch.float32).to(device)
        
        mean_x = self.mean_module(x_transformed)
        covar_x = self.covar_module(x_transformed)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)