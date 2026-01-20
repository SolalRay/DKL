import gpytorch
import torch

# Modele de GPytorch exact GP
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.MaternKernel(nu=2.5)) 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class TransformedGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, function,lengthscale):
        super(TransformedGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.MaternKernel(nu=2.5))
        self.function = function
        self.lengthscale = lengthscale

    def forward(self, x):
        x = self.function(x.cpu().numpy(), scale = self.lengthscale)
        x = torch.tensor(x, dtype=torch.float32)#.to(device)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

