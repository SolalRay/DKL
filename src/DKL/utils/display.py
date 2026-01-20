import numpy as np
import torch
import matplotlib.pyplot as plt
import gpytorch
from tqdm import tqdm
import torch.nn.functional as F
import properscoring as ps

def predict_pointwise_gp(model,likelihood, X_test, batch_size=128, show_progress=True):
    """
    Fait des prédictions point par point ou par petits batches pour éviter
    de calculer la matrice de covariance complète.
    
    Args:
        model: Le modèle GP entraîné
        X_test: Points de test (torch.Tensor)
        batch_size: Taille des batches (1 pour point par point)
        show_progress: Afficher la barre de progression
    
    Returns:
        means: Moyennes prédites
        variances: Variances prédites
    """
    model.eval()
    likelihood.eval()
    
    n_test = X_test.shape[0]
    means = torch.zeros(n_test)
    variances = torch.zeros(n_test)
    
    # Créer un itérateur avec barre de progression si demandé
    if show_progress:
        iterator = tqdm(range(0, n_test, batch_size), 
                       desc="Prédictions GP", 
                       total=int(np.ceil(n_test / batch_size)))
    else:
        iterator = range(0, n_test, batch_size)
    
    with torch.no_grad():
        for i in iterator:
            # Définir les indices du batch
            end_idx = min(i + batch_size, n_test)
            batch_indices = slice(i, end_idx)
            
            # Prédiction sur le batch
            X_batch = X_test[batch_indices]
            pred_dist = model(X_batch)

            # Stocker les résultats
            means[batch_indices] = pred_dist.mean
            variances[batch_indices] = pred_dist.variance
            # Libérer la mémoire
            del pred_dist, X_batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return means, variances




def test_new_realization(trained_flow, trained_gp, trained_likelihood, naive_gp, naive_likelihood, ideal_gp,ideal_likelihood, common_data, realization, scale , variance = False):

    """ Va inferer les Trois GP sur les donnees de test et calculer le MSE """

    print("\n--- Testing with Existing Test Grid ---")
    trained_flow.eval()
    trained_gp.eval()
    trained_likelihood.eval()
    naive_gp.eval()
    naive_likelihood.eval()
    ideal_gp.eval()
    ideal_likelihood.eval()

    print("Using the initial test grid data from generation...")
    X_test_torch_flat = common_data['X_test_grid_torch'] # On utilise les tenseur aplatis
    grid_size = common_data['test_grid_size']
    Y_test_2d_np_true = realization['Y_test_grid_2d']
    Y_test_torch_flat_true = realization['Y_test_grid_torch']

    BATCH_SIZE = 128

    # On commence par le modele principal
    print("Predicting and Sampling from Learned GP on test grid...")
    with torch.no_grad():
        final_trained_z = trained_flow(common_data['X_train_torch'])

        # On transforme les coordonnées de test par le flow
        X_test_learned_z = trained_flow(X_test_torch_flat)
        trained_gp.set_train_data(final_trained_z, realization['Y_train_torch'], strict=True)
        learned_pred_mean_flat, learned_pred_var_flat = predict_pointwise_gp(trained_gp, trained_likelihood, X_test_learned_z, batch_size=BATCH_SIZE, show_progress=True)


    # Pareil pour le Naif
    print("Predicting and Sampling from Guessed GP on test grid...")
    with torch.no_grad():
        guessed_pred_mean_flat, guessed_pred_var_flat = predict_pointwise_gp(naive_gp,naive_likelihood, X_test_torch_flat, batch_size=BATCH_SIZE, show_progress=True)


    # Enfin pour le cas Ideal
    print("Predicting and Sampling from True Transformed GP on test grid...")
    with torch.no_grad():
        true_transformed_pred_mean_flat, true_transformed_pred_var_flat = predict_pointwise_gp(ideal_gp, ideal_likelihood,X_test_torch_flat, batch_size=BATCH_SIZE, show_progress=True)

    # On calcule les MSE
    print("\nCalculating MSE on Initial Test Grid Data...")
    y_true_torch_flat_float = Y_test_torch_flat_true.float().cpu()
    mse_learned_test = F.mse_loss(learned_pred_mean_flat, y_true_torch_flat_float)

    print(f"MSE (Learned Non-Stationary GP) vs True Test Grid (Rzn 0): {mse_learned_test.item():.6f}")

    mse_guessed_test = F.mse_loss(guessed_pred_mean_flat, y_true_torch_flat_float)
    print(f"MSE (Best Stationary GP in Original Space) vs True Test Grid (Rzn 0): {mse_guessed_test.item():.6f}")

    mse_true_transformed_test = F.mse_loss(true_transformed_pred_mean_flat, y_true_torch_flat_float)
    print(f"MSE (Stationary GP on TRUE Transformed Space) vs True Test Grid (Rzn 0): {mse_true_transformed_test.item():.6f}")


    # Visualisation
    print("\n Plotting Predictions Comparison on Initial Test Grid...")
    
    plt.figure(figsize=(16, 12))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)

    fig.suptitle('Comparison of each GP learning results', fontsize=16) # Adjusted title
    plot_extent = [-scale, scale, -scale, scale]

    # Plot 1: Prediction from Learned Non-Stationary GP (Top-Left: axes[0, 0])
    ax = axes[0, 0]
    learned_pred_2d = learned_pred_mean_flat.cpu().numpy().reshape(grid_size, grid_size)
    im = ax.imshow(learned_pred_2d, extent=plot_extent, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Value')
    ax.set_title(f'Learned non-Stationary GP\n(MSE vs True: {mse_learned_test.item():.6f})', fontsize=10)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')


    # Plot 2: Prediction from Best Stationary GP (Original Space) (Top-Right: axes[0, 1])
    ax = axes[0, 1] 
    guessed_pred_2d = guessed_pred_mean_flat.cpu().numpy().reshape(grid_size, grid_size)
    im = ax.imshow(guessed_pred_2d, extent=plot_extent, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Value') 
    ax.set_title(f'Stationary GP \n(MSE vs True: {mse_guessed_test.item():.6f})', fontsize=10)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')


    # Plot 3: Prediction from Stationary GP on TRUE Transformed Space (Bottom-Left: axes[1, 0])
    ax = axes[1, 0]
    true_transformed_pred_2d = true_transformed_pred_mean_flat.cpu().numpy().reshape(grid_size, grid_size)
    im = ax.imshow(true_transformed_pred_2d, extent=plot_extent, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Value')
    ax.set_title(f'Transformed GP\n(MSE vs True: {mse_true_transformed_test.item():.6f})', fontsize=10)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')


    # Plot 4: True Test Grid Realization (from Realization 0) (Bottom-Right: axes[1, 1])
    ax = axes[1, 1] 
    im = ax.imshow(Y_test_2d_np_true, extent=plot_extent, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Value') # Add colorbar associated with this specific axes
    ax.set_title(f'True Test Data', fontsize=10) # Simplified title
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("\n Plotting Variances Comparison on Initial Test Grid...")
    
    plt.figure(figsize=(16, 12))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)

    fig.suptitle('Comparison of each GP learning results', fontsize=16) # Adjusted title
    plot_extent = [-scale, scale, -scale, scale]

    # Plot 1: Prediction variance from Learned Non-Stationary GP (Top-Left: axes[0, 0])
    ax = axes[0, 0]
    learned_var_2d = learned_pred_var_flat.cpu().numpy().reshape(grid_size, grid_size)
    im = ax.imshow(learned_var_2d, extent=plot_extent, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Value')
    ax.set_title(f'Learned non-Stationary GP\n(MSE vs True: {mse_learned_test.item():.6f})', fontsize=10)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')


    # Plot 2: Prediction variance from Best Stationary GP (Original Space) (Top-Right: axes[0, 1])
    ax = axes[0, 1] 
    guessed_var_2d = guessed_pred_var_flat.cpu().numpy().reshape(grid_size, grid_size)
    im = ax.imshow(guessed_var_2d, extent=plot_extent, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Value') 
    ax.set_title(f'Stationary GP \n(MSE vs True: {mse_guessed_test.item():.6f})', fontsize=10)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')


    # Plot 3: Prediction variance from Stationary GP on TRUE Transformed Space (Bottom-Left: axes[1, 0])
    ax = axes[1, 0]
    true_transformed_var_2d = true_transformed_pred_var_flat.cpu().numpy().reshape(grid_size, grid_size)
    im = ax.imshow(true_transformed_var_2d, extent=plot_extent, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Value')
    ax.set_title(f'Transformed GP\n(MSE vs True: {mse_true_transformed_test.item():.6f})', fontsize=10)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')


    # Plot 4: True Test Grid Realization (from Realization 0) (Bottom-Right: axes[1, 1])
    ax = axes[1, 1] 
    im = ax.imshow(Y_test_2d_np_true, extent=plot_extent, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Value') # Add colorbar associated with this specific axes
    ax.set_title(f'True Test Data', fontsize=10) # Simplified title
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    print("\n Plotting CRPS Comparison on Initial Test Grid...")
    
    plt.figure(figsize=(16, 12))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)

    fig.suptitle('Comparison of each GP learning results', fontsize=16) # Adjusted title
    plot_extent = [-scale, scale, -scale, scale]

    # Plot 1: Prediction variance from Learned Non-Stationary GP (Top-Left: axes[0, 0])
    ax = axes[0, 0]
    crps_learned = ps.crps_gaussian(Y_test_2d_np_true,learned_pred_2d, np.sqrt(learned_var_2d))
    im = ax.imshow(crps_learned, extent=plot_extent, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Value')
    ax.set_title(f'Learned non-Stationary GP\n(MSE vs True: {mse_learned_test.item():.6f})', fontsize=10)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')


    # Plot 2: Prediction variance from Best Stationary GP (Original Space) (Top-Right: axes[0, 1])
    ax = axes[0, 1] 
    crps_guessed = ps.crps_gaussian(Y_test_2d_np_true,guessed_pred_2d, np.sqrt(guessed_var_2d))
    im = ax.imshow(crps_guessed, extent=plot_extent, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Value') 
    ax.set_title(f'Stationary GP \n(MSE vs True: {mse_guessed_test.item():.6f})', fontsize=10)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')


    # Plot 3: Prediction variance from Stationary GP on TRUE Transformed Space (Bottom-Left: axes[1, 0])
    ax = axes[1, 0]
    crps_true_transformed = ps.crps_gaussian(Y_test_2d_np_true,true_transformed_pred_2d, np.sqrt(true_transformed_var_2d))
    im = ax.imshow(crps_true_transformed, extent=plot_extent, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Value')
    ax.set_title(f'Transformed GP\n(MSE vs True: {mse_true_transformed_test.item():.6f})', fontsize=10)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')


    # Plot 4: True Test Grid Realization (from Realization 0) (Bottom-Right: axes[1, 1])
    ax = axes[1, 1] 
    im = ax.imshow(Y_test_2d_np_true, extent=plot_extent, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Value') # Add colorbar associated with this specific axes
    ax.set_title(f'True Test Data', fontsize=10) # Simplified title
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # boxplots of the CRPS
    plt.figure(figsize=(16, 12))
    plt.boxplot([crps_learned.flatten(), crps_guessed.flatten(), crps_true_transformed.flatten()], tick_labels=['Learned', 'Guessed', 'True Transformed'])
    plt.title('CRPS Comparison')
    plt.ylabel('CRPS')
    plt.yscale('log')
    plt.grid()
    plt.show()

    print("Evaluation finished.")


def plotting(trained_flow,trained_gp,trained_likelihood,common_data, realization,fonction,loss_history,scale):
    trained_flow.eval()
    trained_gp.eval() # Ensure eval mode
    trained_likelihood.eval()


    with torch.no_grad():
        learned_transformed_z = trained_flow(common_data['X_train_torch'])
        learned_transformed_z_np = learned_transformed_z.cpu().numpy()
        X_np = common_data['X_train_np']
        X_transformed_true_np = fonction(X_np, scale)

    # Plot the training loss (same as before)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss : Negative Marginal Log-Likelihood")
    plt.title("Joint Training Loss")
    plt.grid(True)
    plt.show()

    # Plot the original data points and learned transformed points (scatter plots, same as before)
    first_realization_y_np = realization['Y_train_np'] # Use the first realization for color coding
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(X_np[:, 0], X_np[:, 1], c=first_realization_y_np, cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(scatter1)
    plt.title('Original Data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.axis('equal')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(learned_transformed_z_np[:, 0], learned_transformed_z_np[:, 1], c=first_realization_y_np, cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(scatter2)
    plt.title('Original Points in Learned Transformed Space')
    plt.xlabel('Learned Z1')
    plt.ylabel('Learned Z2')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # Optional: Plot the true transformed points vs learned transformed points (scatter plots, same as before)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    scatter3 = plt.scatter(X_transformed_true_np[:, 0], X_transformed_true_np[:, 1], c=first_realization_y_np, cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(scatter3)
    plt.title('Original Points in True Transformed Space')
    plt.xlabel('True Transformed X1')
    plt.ylabel('True Transformed X2')
    plt.axis('equal')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    scatter4 = plt.scatter(learned_transformed_z_np[:, 0], learned_transformed_z_np[:, 1], c=first_realization_y_np, cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(scatter4)
    plt.title('Original Points in Learned Transformed Space')
    plt.xlabel('Learned Z1')
    plt.ylabel('Learned Z2')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # Optional: Visualize the learned transformation grid (same as before)
    print("\nGenerating visualization grid for learned transformation...")
    grid_steps_viz = 20
    x_viz = np.linspace(-scale, scale, grid_steps_viz)
    y_viz = np.linspace(-scale, scale, grid_steps_viz)
    X_viz_grid_np, Y_viz_grid_np = np.meshgrid(x_viz, y_viz)
    X_viz_np = np.vstack([X_viz_grid_np.ravel(), Y_viz_grid_np.ravel()]).T
    X_viz_torch = torch.tensor(X_viz_np, dtype=torch.float32)#.to(device)

    with torch.no_grad():
        learned_viz_z = trained_flow(X_viz_torch)
        learned_viz_z_np = learned_viz_z.cpu().numpy()

    Z_viz_grid_x = learned_viz_z_np[:, 0].reshape(grid_steps_viz, grid_steps_viz)
    Z_viz_grid_y = learned_viz_z_np[:, 1].reshape(grid_steps_viz, grid_steps_viz)

    plt.figure(figsize=(7, 7))
    for i in range(grid_steps_viz):
        plt.plot(Z_viz_grid_x[i, :], Z_viz_grid_y[i, :], color='gray', alpha=0.5)
        plt.plot(Z_viz_grid_x[:, i], Z_viz_grid_y[:, i], color='gray', alpha=0.5)

    plt.title('Learned Transformation of Original Grid')
    plt.xlabel('Learned Z1')
    plt.ylabel('Learned Z2')
    plt.axis('equal')
    plt.grid(True)
    plt.show()
    #vis.final(trained_flow)
