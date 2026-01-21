import os
import torch
import gpytorch
import torch.optim as optim
from tqdm import tqdm
from ..models.kernel import ExactGPModel
from ..models.kernel import TransformedGPModel
from ..training.early_stopping import EarlyStopping

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_naive_gp(common_data, realization, num_epochs=100, lr=0.1, 
                   checkpoint_name='naive_gp_checkpoint.pth', 
                   device = "cpu",
                   patience=10, 
                   delta=0):     
    """
    Trains the Naive GP with Early Stopping.
    Saves the model only if there is improvement and the epoch is a multiple of 10.
    """
    checkpoint_path = MODELS_DIR / checkpoint_name
    X_train_torch = common_data['X_train_torch'].to(device)
    Y_train_torch = realization['Y_train_torch'].to(device)

    naive_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    naive_gp_model = ExactGPModel(X_train_torch, Y_train_torch, naive_likelihood).to(device)

    naive_optimizer = optim.Adam([
        {'params': naive_gp_model.parameters()},
    ], lr=lr)

    mll_naive = gpytorch.mlls.ExactMarginalLogLikelihood(naive_likelihood, naive_gp_model)

    start_epoch = 0
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    if os.path.exists(checkpoint_path):
        print(f"Loading Naive GP checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        naive_gp_model.load_state_dict(checkpoint['gp_model_state_dict'])
        naive_likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        naive_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming Naive GP training from epoch {start_epoch}")
    else:
        print("No Naive GP checkpoint found. Starting training from scratch.")

    naive_gp_model.train()
    naive_likelihood.train()

    print(f"Training Naive GP on original data for {num_epochs} epochs (starting at {start_epoch})...")
    for k in tqdm(range(start_epoch, num_epochs)):
        naive_optimizer.zero_grad()
        output_naive = naive_gp_model(X_train_torch)
        loss_naive = -mll_naive(output_naive, Y_train_torch)
        loss_naive.backward()
        naive_optimizer.step()

        current_loss_item = loss_naive.item()

        # Early Stopping and checkpointing logic
        improved = early_stopping(current_loss_item)

        if improved and (k + 1) % 10 == 0: 
            print(f'  Naive GP Epoch {k+1}/{num_epochs}: Loss {current_loss_item:.4f}, Lengthscale: {naive_gp_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()}, Outputscale: {naive_gp_model.covar_module.outputscale.item():.4f}, Noise: {naive_likelihood.noise.item():.4f}')
            print(f"  Best loss detected and epoch eligible for saving.")
            torch.save({
                'epoch': k,
                'gp_model_state_dict': naive_gp_model.state_dict(),
                'likelihood_state_dict': naive_likelihood.state_dict(),
                'optimizer_state_dict': naive_optimizer.state_dict(),
                'val_loss_min': early_stopping.val_loss_min,
                'best_score': early_stopping.best_score,
            }, checkpoint_path)

        if early_stopping.early_stop:
            print(f"Early Stopping triggered at epoch {k+1}! No improvement for {patience} epochs.")
            break 

    print("Naive GP training complete.")

    # Load the BEST model saved by Early Stopping
    if os.path.exists(checkpoint_path):
        print(f"Loading best Naive GP model from {checkpoint_path} for final return...")
        best_checkpoint = torch.load(checkpoint_path, map_location=device)
        naive_gp_model.load_state_dict(best_checkpoint['gp_model_state_dict'])
        naive_likelihood.load_state_dict(best_checkpoint['likelihood_state_dict'])
    else:
        print(f"Warning: No best model checkpoint found at {checkpoint_path}. "
              "Returning model from the last trained epoch.")

    naive_gp_model.eval()
    naive_likelihood.eval()

    return naive_gp_model, naive_likelihood


def train_ideal_gp(common_data, realization, function,
                   num_epochs=100, lr=0.1,
                   lengthscale=2.0,
                   device = "cpu",
                   checkpoint_name='ideal_gp_checkpoint.pth', 
                   patience=10, 
                   delta=0):     
    """
    Trains the Ideal GP on transformed data with Early Stopping.
    Saves the model only if there is improvement and the epoch is a multiple of 10.
    """
    X_train_torch = common_data['X_train_torch'].to(device)
    Y_train_torch = realization['Y_train_torch'].to(device)
    checkpoint_path = MODELS_DIR / checkpoint_name
    
    ideal_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    # Ensure TransformedGPModel is correctly defined in your kernel module
    ideal_gp_model = TransformedGPModel(X_train_torch, Y_train_torch, ideal_likelihood, function, lengthscale).to(device) 

    ideal_optimizer = optim.Adam([
        {'params': ideal_gp_model.parameters()},
    ], lr=lr)

    mll_ideal = gpytorch.mlls.ExactMarginalLogLikelihood(ideal_likelihood, ideal_gp_model)

    start_epoch = 0
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    if os.path.exists(checkpoint_path):
        print(f"Loading Transformed GP checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ideal_gp_model.load_state_dict(checkpoint['gp_model_state_dict'])
        ideal_likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        ideal_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming Transformed GP training from epoch {start_epoch}")
    else:
        print("No Transformed GP checkpoint found. Starting training from scratch.")

    ideal_gp_model.train()
    ideal_likelihood.train()

    print(f"Training Transformed GP for {num_epochs} epochs (starting at {start_epoch})...")
    for k in tqdm(range(start_epoch, num_epochs)):
        ideal_optimizer.zero_grad()
        output_ideal = ideal_gp_model(X_train_torch)
        loss_ideal = -mll_ideal(output_ideal, Y_train_torch)
        loss_ideal.backward()
        ideal_optimizer.step()

        current_loss_item = loss_ideal.item()

        # Early Stopping and checkpointing logic
        improved = early_stopping(current_loss_item)

        if improved and (k + 1) % 10 == 0: 
            print(f'  Transformed GP Epoch {k+1}/{num_epochs}: Loss {current_loss_item:.4f}, Lengthscale: {ideal_gp_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()}, Outputscale: {ideal_gp_model.covar_module.outputscale.item():.4f}, Noise: {ideal_likelihood.noise.item():.4f}')
            print(f"  Best loss detected and epoch eligible for saving.")
            torch.save({
                'epoch': k,
                'gp_model_state_dict': ideal_gp_model.state_dict(),
                'likelihood_state_dict': ideal_likelihood.state_dict(),
                'optimizer_state_dict': ideal_optimizer.state_dict(),
                'val_loss_min': early_stopping.val_loss_min,
                'best_score': early_stopping.best_score,
            }, checkpoint_path)

        if early_stopping.early_stop:
            print(f"Early Stopping triggered at epoch {k+1}! No improvement for {patience} epochs.")
            break 

    print("Transformed GP training complete.")

    # Load the BEST model saved by Early Stopping
    if os.path.exists(checkpoint_path):
        print(f"Loading best Ideal GP model from {checkpoint_path} for final return...")
        best_checkpoint = torch.load(checkpoint_path, map_location=device)
        ideal_gp_model.load_state_dict(best_checkpoint['gp_model_state_dict'])
        ideal_likelihood.load_state_dict(best_checkpoint['likelihood_state_dict'])
    else:
        print(f"Warning: No best model checkpoint found at {checkpoint_path}. "
              "Returning model from the last trained epoch.")

    ideal_gp_model.eval()
    ideal_likelihood.eval()

    return ideal_gp_model, ideal_likelihood