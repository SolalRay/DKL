import numpy as np
import torch
import gstlearn as gl

def modelgpytorch(nu, scale):
    """Creates a Matern covariance model using gstlearn compatible with gpytorch's Matern kernel."""
    return gl.Model.createFromParam(gl.ECov.MATERN, param=nu, range=1./np.sqrt(2 * nu) * scale, flagRange=False)

def generation_approchee(x, y, nu, scale, noise):
    db = gl.Db.create()
    db["x"] = x
    db["y"] = y
    db.setLocators(["x", "y"], gl.ELoc.X)
    model = modelgpytorch(nu, scale)
    gl.simtub(None, db, model, nbtuba=10000)
    return db["Simu"] 

def generate_2d_data(transformation, n_train_points=1000, test_grid_size=100, scale=5.,
                       lengthscale=2.0, nu=2.5, noise_level=0.05,
                       base_seed=None, device='cpu'):
    """
    Generates a realization of a Gaussian Process on random points for training
    and on a regular grid for testing. The GP is generated on the union of training and test points. 
    Note: The GP generation occurs in the transformed space defined by the function in the module.
    """
    print(f"  - {n_train_points} random training points per realization")
    print(f"  - {test_grid_size}x{test_grid_size} visualization grid ({test_grid_size*test_grid_size} points) for testing")

    if base_seed is not None:
        master_rng = np.random.RandomState(base_seed)
    else:
        master_rng = np.random.RandomState()

    # Generate training points
    X_train_np = master_rng.uniform(-scale, scale, (n_train_points, 2))  # 2D numpy points
    X_train_torch = torch.tensor(X_train_np, dtype=torch.float32).to(device)   # torch conversion

    # Generate test points
    x_grid = np.linspace(-scale, scale, test_grid_size)
    y_grid = np.linspace(-scale, scale, test_grid_size)
    
    X_test_grid_grid_np, Y_test_grid_grid_np = np.meshgrid(x_grid, y_grid)
    X_test_grid_np = np.vstack([X_test_grid_grid_np.ravel(), Y_test_grid_grid_np.ravel()]).T
    X_test_grid_torch = torch.tensor(X_test_grid_np, dtype=torch.float32).to(device)

    # Stack points for GP generation, then transform via the function defined in the module
    X_combined_np = np.vstack([X_train_np, X_test_grid_np])
    X_combined_transformed_np_for_gen = transformation(X_combined_np, scale=lengthscale)

    # Store common data (points and grid/scale parameters)
    common_data = {
        'X_train_np': X_train_np, # Training points NumPy
        'X_train_torch': X_train_torch, # Training points Torch
        'X_test_grid_np': X_test_grid_np, # Flattened grid points NumPy
        'X_test_grid_torch': X_test_grid_torch, # Flattened grid points Torch
        'X_test_grid_grid_np': X_test_grid_grid_np, # Meshgrid X points NumPy
        'Y_test_grid_grid_np': Y_test_grid_grid_np, # Meshgrid Y points NumPy
        'test_grid_size': test_grid_size,
        'n_train_points': n_train_points,
        'scale': scale,
    }
    
    # Generate a Gaussian Process realization using gstlearn on both training and test points
    y_combined_np = generation_approchee(X_combined_transformed_np_for_gen[:,0],
                                         X_combined_transformed_np_for_gen[:,1],
                                         nu, scale, noise_level)

    # Separate training and test data
    # Add noise to training points
    y_train_np = y_combined_np[:n_train_points] + np.sqrt(noise_level) * np.random.normal(size=n_train_points) 
    
    # The remainder is for the test grid
    y_test_grid_np = y_combined_np[n_train_points:] 
    
    y_train_torch = torch.tensor(y_train_np, dtype=torch.float32).to(device)
    y_test_grid_torch = torch.tensor(y_test_grid_np, dtype=torch.float32).to(device)

    # Reshape grid values to 2D for potential direct visualization
    y_test_grid_2d = y_test_grid_np.reshape(test_grid_size, test_grid_size)

    # Store data for this specific realization
    realization = {
        'Y_train_np': y_train_np, # Target y at training points (np)
        'Y_train_torch': y_train_torch, # Target y at training points (torch)
        'Y_test_grid_np': y_test_grid_np, # Target y at test grid points (flattened np)
        'Y_test_grid_torch': y_test_grid_torch, # Target y at test grid points (flattened torch)
        'Y_test_grid_2d': y_test_grid_2d, # Target y at test grid points (2D grid np)
    }
    
    print(f"Y Stats (train) min={y_train_np.min():.4f}, max={y_train_np.max():.4f}, mean={y_train_np.mean():.4f}, std={y_train_np.std():.4f}")
    print(f"Y Stats (test grid) min={y_test_grid_np.min():.4f}, max={y_test_grid_np.max():.4f}, mean={y_test_grid_np.mean():.4f}, std={y_test_grid_np.std():.4f}")

    print("Data generation complete.")

    return common_data, realization