import numpy as np
import torch
import matplotlib.pyplot as plt
import gpytorch
from tqdm import tqdm
import torch.nn.functional as F
import properscoring as ps

def predict_pointwise_gp(model, likelihood, X_test, batch_size=128, show_progress=True):
    """
    Performs predictions point-by-point or in small batches to avoid
    computing the full covariance matrix (memory efficiency).
    
    Args:
        model: Trained GP model
        X_test: Test points (torch.Tensor)
        batch_size: Batch size (use 1 for point-by-point)
        show_progress: Whether to display a progress bar
    
    Returns:
        means: Predicted means
        variances: Predicted variances
    """
    model.eval()
    likelihood.eval()
    
    n_test = X_test.shape[0]
    means = torch.zeros(n_test)
    variances = torch.zeros(n_test)
    
    # Create iterator with progress bar if requested
    if show_progress:
        iterator = tqdm(range(0, n_test, batch_size), 
                       desc="GP Predictions", 
                       total=int(np.ceil(n_test / batch_size)))
    else:
        iterator = range(0, n_test, batch_size)
    
    with torch.no_grad():
        for i in iterator:
            # Define batch indices
            end_idx = min(i + batch_size, n_test)
            batch_indices = slice(i, end_idx)
            
            # Predict on the batch
            X_batch = X_test[batch_indices]
            pred_dist = model(X_batch)

            # Store results
            means[batch_indices] = pred_dist.mean
            variances[batch_indices] = pred_dist.variance
            
            # Memory cleanup
            del pred_dist, X_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return means, variances


def test_new_realization(trained_flow, trained_gp, trained_likelihood, naive_gp, naive_likelihood, ideal_gp, ideal_likelihood, common_data, realization, scale, variance=False):
    """
    Infers the three GPs on test data and calculates MSE, Variances, and CRPS.
    """

    print("\n--- Testing with Existing Test Grid ---")
    # Set all models to evaluation mode
    trained_flow.eval()
    trained_gp.eval()
    trained_likelihood.eval()
    naive_gp.eval()
    naive_likelihood.eval()
    ideal_gp.eval()
    ideal_likelihood.eval()

    print("Using the initial test grid data from generation...")
    X_test_torch_flat = common_data['X_test_grid_torch'] # Use flattened tensors
    grid_size = common_data['test_grid_size']
    Y_test_2d_np_true = realization['Y_test_grid_2d']
    Y_test_torch_flat_true = realization['Y_test_grid_torch']

    BATCH_SIZE = 128

    # 1. Main Learned Model (DKL)
    print("Predicting and Sampling from Learned GP on test grid...")
    with torch.no_grad():
        final_trained_z = trained_flow(common_data['X_train_torch'])

        # Transform test coordinates through the flow
        X_test_learned_z = trained_flow(X_test_torch_flat)
        trained_gp.set_train_data(final_trained_z, realization['Y_train_torch'], strict=True)
        learned_pred_mean_flat, learned_pred_var_flat = predict_pointwise_gp(trained_gp, trained_likelihood, X_test_learned_z, batch_size=BATCH_SIZE, show_progress=True)

    # 2. Naive Model (Stationary GP)
    print("Predicting and Sampling from Guessed (Naive) GP on test grid...")
    with torch.no_grad():
        guessed_pred_mean_flat, guessed_pred_var_flat = predict_pointwise_gp(naive_gp, naive_likelihood, X_test_torch_flat, batch_size=BATCH_SIZE, show_progress=True)

    # 3. Ideal Case (Stationary GP on Ground Truth Transformation)
    print("Predicting and Sampling from True Transformed (Ideal) GP on test grid...")
    with torch.no_grad():
        true_transformed_pred_mean_flat, true_transformed_pred_var_flat = predict_pointwise_gp(ideal_gp, ideal_likelihood, X_test_torch_flat, batch_size=BATCH_SIZE, show_progress=True)

    # MSE Calculations
    print("\nCalculating MSE on Initial Test Grid Data...")
    y_true_torch_flat_float = Y_test_torch_flat_true.float().cpu()
    
    mse_learned_test = F.mse_loss(learned_pred_mean_flat, y_true_torch_flat_float)
    print(f"MSE (Learned Non-Stationary GP) vs True Test Grid: {mse_learned_test.item():.6f}")

    mse_guessed_test = F.mse_loss(guessed_pred_mean_flat, y_true_torch_flat_float)
    print(f"MSE (Best Stationary GP in Original Space) vs True Test Grid: {mse_guessed_test.item():.6f}")

    mse_true_transformed_test = F.mse_loss(true_transformed_pred_mean_flat, y_true_torch_flat_float)
    print(f"MSE (Stationary GP on TRUE Transformed Space) vs True Test Grid: {mse_true_transformed_test.item():.6f}")

    # Visualization: Predictions
    print("\nPlotting Predictions Comparison on Initial Test Grid...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle('Comparison of Prediction Results for each GP', fontsize=16)
    plot_extent = [-scale, scale, -scale, scale]

    # Helper for reshaping results
    learned_pred_2d = learned_pred_mean_flat.cpu().numpy().reshape(grid_size, grid_size)
    guessed_pred_2d = guessed_pred_mean_flat.cpu().numpy().reshape(grid_size, grid_size)
    true_transformed_pred_2d = true_transformed_pred_mean_flat.cpu().numpy().reshape(grid_size, grid_size)

    # Subplots for Predictions
    results = [
        (learned_pred_2d, f'Learned non-Stationary GP\n(MSE: {mse_learned_test.item():.6f})'),
        (guessed_pred_2d, f'Stationary GP\n(MSE: {mse_guessed_test.item():.6f})'),
        (true_transformed_pred_2d, f'Ideal Transformed GP\n(MSE: {mse_true_transformed_test.item():.6f})'),
        (Y_test_2d_np_true, 'True Test Data')
    ]

    for ax, (data, title) in zip(axes.flat, results):
        im = ax.imshow(data, extent=plot_extent, origin='lower', cmap='viridis')
        fig.colorbar(im, ax=ax, label='Value')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Visualization: Variances
    print("\nPlotting Variances Comparison on Initial Test Grid...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle('Comparison of Uncertainty (Variance) for each GP', fontsize=16)

    learned_var_2d = learned_pred_var_flat.cpu().numpy().reshape(grid_size, grid_size)
    guessed_var_2d = guessed_pred_var_flat.cpu().numpy().reshape(grid_size, grid_size)
    true_transformed_var_2d = true_transformed_pred_var_flat.cpu().numpy().reshape(grid_size, grid_size)

    vars_to_plot = [
        (learned_var_2d, 'Learned GP Variance'),
        (guessed_var_2d, 'Stationary GP Variance'),
        (true_transformed_var_2d, 'Ideal GP Variance'),
        (Y_test_2d_np_true, 'True Test Data (Reference)')
    ]

    for ax, (data, title) in zip(axes.flat, vars_to_plot):
        im = ax.imshow(data, extent=plot_extent, origin='lower', cmap='magma')
        fig.colorbar(im, ax=ax, label='Uncertainty')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Visualization: CRPS (Continuous Ranked Probability Score)
    print("\nPlotting CRPS Comparison on Initial Test Grid...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle('Comparison of CRPS (Local Prediction Quality)', fontsize=16)

    crps_learned = ps.crps_gaussian(Y_test_2d_np_true, learned_pred_2d, np.sqrt(learned_var_2d))
    crps_guessed = ps.crps_gaussian(Y_test_2d_np_true, guessed_pred_2d, np.sqrt(guessed_var_2d))
    crps_true_transformed = ps.crps_gaussian(Y_test_2d_np_true, true_transformed_pred_2d, np.sqrt(true_transformed_var_2d))

    crps_results = [
        (crps_learned, 'Learned GP CRPS'),
        (crps_guessed, 'Stationary GP CRPS'),
        (crps_true_transformed, 'Ideal GP CRPS'),
        (Y_test_2d_np_true, 'True Test Data (Reference)')
    ]

    for ax, (data, title) in zip(axes.flat, crps_results):
        im = ax.imshow(data, extent=plot_extent, origin='lower', cmap='inferno')
        fig.colorbar(im, ax=ax, label='CRPS Score')
        ax.set_title(title, fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Statistical Comparison: CRPS Boxplots
    plt.figure(figsize=(10, 6))
    plt.boxplot([crps_learned.flatten(), crps_guessed.flatten(), crps_true_transformed.flatten()], 
                tick_labels=['Learned DKL', 'Naive Stationary', 'Ideal Transformed'])
    plt.title('CRPS Distribution Comparison')
    plt.ylabel('CRPS (lower is better)')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()

    print("Evaluation finished.")


def plotting(trained_flow, trained_gp, trained_likelihood, common_data, realization, function, loss_history, scale):
    """
    Plots training loss and visualizes the warping/deformation learned by the flow.
    """
    trained_flow.eval()
    trained_gp.eval() 
    trained_likelihood.eval()

    device = common_data['X_train_torch'].device

    with torch.no_grad():
        learned_transformed_z = trained_flow(common_data['X_train_torch'])
        learned_transformed_z_np = learned_transformed_z.cpu().numpy()
        X_np = common_data['X_train_np']
        X_transformed_true_np = function(X_np, scale)

    # 1. Training Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Marginal Log-Likelihood")
    plt.title("Joint Training Loss (Flow + GP)")
    plt.grid(True)
    plt.show()

    # 2. Comparison: Original Space vs Learned Latent Space
    y_train_np = realization['Y_train_np']
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_train_np, cmap='viridis', s=10, alpha=0.6)
    plt.title('Original Data Space')
    plt.axis('equal')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(learned_transformed_z_np[:, 0], learned_transformed_z_np[:, 1], c=y_train_np, cmap='viridis', s=10, alpha=0.6)
    plt.title('Data in Learned Latent Space (Z)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # 3. Comparison: True Transformed Space vs Learned Transformed Space
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X_transformed_true_np[:, 0], X_transformed_true_np[:, 1], c=y_train_np, cmap='viridis', s=10, alpha=0.6)
    plt.title('Ground Truth Transformed Space')
    plt.axis('equal')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(learned_transformed_z_np[:, 0], learned_transformed_z_np[:, 1], c=y_train_np, cmap='viridis', s=10, alpha=0.6)
    plt.title('Learned Transformed Space')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # 4. Grid Distortion Visualization
    print("\nGenerating visualization grid for learned transformation...")
    grid_steps_viz = 20
    x_viz = np.linspace(-scale, scale, grid_steps_viz)
    y_viz = np.linspace(-scale, scale, grid_steps_viz)
    X_viz_grid_np, Y_viz_grid_np = np.meshgrid(x_viz, y_viz)
    X_viz_np = np.vstack([X_viz_grid_np.ravel(), Y_viz_grid_np.ravel()]).T
    X_viz_torch = torch.tensor(X_viz_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        learned_viz_z = trained_flow(X_viz_torch)
        learned_viz_z_np = learned_viz_z.cpu().numpy()

    Z_viz_grid_x = learned_viz_z_np[:, 0].reshape(grid_steps_viz, grid_steps_viz)
    Z_viz_grid_y = learned_viz_z_np[:, 1].reshape(grid_steps_viz, grid_steps_viz)

    plt.figure(figsize=(7, 7))
    for i in range(grid_steps_viz):
        plt.plot(Z_viz_grid_x[i, :], Z_viz_grid_y[i, :], color='gray', alpha=0.5)
        plt.plot(Z_viz_grid_x[:, i], Z_viz_grid_y[:, i], color='gray', alpha=0.5)

    plt.title('Visualization of the Learned Spatial Warping')
    plt.xlabel('Learned Z1')
    plt.ylabel('Learned Z2')
    plt.axis('equal')
    plt.grid(True)
    plt.show()