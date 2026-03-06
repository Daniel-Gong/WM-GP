import torch
import gpytorch

class WorkingMemoryGP(gpytorch.models.ApproximateGP):
    """
    A unified 2D Gaussian Process model for Visual Working Memory.
    Input dims: [Location (-180 to 180), Color (-180 to 180)]
    """
    def __init__(self, inducing_grid_size: int = 36, loc_lengthscale: float = 10.0, color_lengthscale: float = 10.0, learn_inducing_locations: bool = True):
        # Initialize inducing points on a inducing_grid_size x inducing_grid_size regular grid over [-180, 180).
        # Offset by half the bin width (180 / inducing_grid_size) so each point sits at the
        # centre of its bin and the distance to each edge equals half the inter-point spacing.
        half_spacing = 180.0 / inducing_grid_size
        grid_1d = torch.linspace(-180.0 + half_spacing, 180.0 - half_spacing, inducing_grid_size)
        grid_loc, grid_color = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
        inducing_points = torch.stack([grid_loc.reshape(-1), grid_color.reshape(-1)], dim=1)  # (inducing_grid_size^2, 2)
        num_inducing_points = inducing_points.shape[0]  # inducing_grid_size * inducing_grid_size
        
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, 
            inducing_points, 
            variational_distribution, 
            learn_inducing_locations=learn_inducing_locations
        )
        super().__init__(variational_strategy)
        
        # Mean module
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Periodic Covariance module for Location (dim 0)
        self.covar_module_loc = gpytorch.kernels.PeriodicKernel(active_dims=torch.tensor([0]))
        self.covar_module_loc.period_length = 360.0  # Fixed period for circle
        self.covar_module_loc.lengthscale = loc_lengthscale
        
        # Periodic Covariance module for Color (dim 1)
        self.covar_module_color = gpytorch.kernels.PeriodicKernel(active_dims=torch.tensor([1]))
        self.covar_module_color.period_length = 360.0 # Fixed period for circle
        self.covar_module_color.lengthscale = color_lengthscale
        
        # Scale kernel wraps the multiplicative composition
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.covar_module_loc * self.covar_module_color
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_inducing_points(self):
        """Return current inducing point locations."""
        return self.variational_strategy.inducing_points
    
    def get_inducing_values(self):
        """Return current variational mean (inducing values)."""
        return self.variational_strategy.variational_distribution.mean
