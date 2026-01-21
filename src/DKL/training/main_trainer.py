import os
import torch
import torch.optim as optim
import gpytorch
from DKL.models.normalizing_flow import RealNVP
from DKL.models.kernel import ExactGPModel
from DKL.training.early_stopping import EarlyStopping
from tqdm import tqdm

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_joint_model(common_data, realization, num_epochs=500, flow_lr=0.001, gp_lr=0.01, num_flow_blocks=12, no_learn_lengthscale=False,
                      checkpoint_name='learned_gp_checkpoint.pth', # Path for periodic checkpoints
                      device = "cpu",
                      patience=10, # Patience for Early Stopping
                      delta=0     # Delta for Early Stopping
                      ): 
    """
    Trains the flow and the GP simultaneously with Early Stopping.
    Uses differentiated learning rates. Option to freeze the GP lengthscale.
    Allows resuming training from a checkpoint.
    """

    flow_model = RealNVP(num_blocks=num_flow_blocks).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    checkpoint_path = MODELS_DIR / checkpoint_name

    # Initialize GP model with dummy data
    dummy_train_x = torch.zeros_like(common_data['X_train_torch']).to(device)
    dummy_train_y = torch.zeros_like(realization['Y_train_torch']).to(device)
    gp_model = ExactGPModel(dummy_train_x, dummy_train_y, likelihood).to(device)

    # Freezing the Lengthscale
    if no_learn_lengthscale:
        print("\nFreezing the lengthscale...")
        gp_model.covar_module.base_kernel.raw_lengthscale.requires_grad = False

    # Separate different parameters
    param_groups = [
        {'params': flow_model.parameters(), 'lr': flow_lr},
    ]
    
    gp_learnable_params = []
    if not no_learn_lengthscale:
        # Add trainable GP parameters if available
        gp_learnable_params = list(filter(lambda p: p.requires_grad, gp_model.parameters()))
    
    if gp_learnable_params:
        param_groups.append({'params': gp_learnable_params, 'lr': gp_lr})

    optimizer = optim.Adam(param_groups)

    # MLL is the loss function for GP
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    start_epoch = 0
    loss_history = []
    
    # --- EarlyStopping Initialization ---
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    # If resuming training from a given point
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        flow_model.load_state_dict(checkpoint['flow_model_state_dict'])
        gp_model.load_state_dict(checkpoint['gp_model_state_dict'])
        likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss_history = checkpoint['loss_history']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    flow_model.train()
    gp_model.train()
    likelihood.train()

    print("\nStarting training...")

    for i in tqdm(range(start_epoch, num_epochs)): # Loop starting from start_epoch
        optimizer.zero_grad()

        Y_train_torch = realization['Y_train_torch'].to(device) # Ensure data is on the correct device
        X_train_torch = common_data['X_train_torch'].to(device) # Ensure data is on the correct device

        # Transform coordinates through the Normalizing Flow (NF)
        transformed_z = flow_model(X_train_torch)

        # Train the GP on the NF output
        gp_model.set_train_data(transformed_z.detach(), Y_train_torch, strict=False)
        output_distribution = gp_model(transformed_z)

        # Calculate loss (Negative Log Likelihood)
        loss = -mll(output_distribution, Y_train_torch)

        # Perform backward pass
        loss.backward()

        ## Gradient clipping (useful when freezing lengthscale or handling deep flows)
        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
        if not no_learn_lengthscale:
             gp_learnable_params_now = list(filter(lambda p: p.requires_grad and p.grad is not None, gp_model.parameters()))
             if gp_learnable_params_now:
                  torch.nn.utils.clip_grad_norm_(gp_learnable_params_now, max_norm=1.0)

        optimizer.step()
        loss_history.append(loss.item())

        current_total_loss = loss.item()

        # Check for improvement via Early Stopping
        improved = early_stopping(current_total_loss)

        # Save checkpoint if loss improved
        if improved and (i + 1) % 10:
            torch.save({
                'epoch': i,
                'flow_model_state_dict': flow_model.state_dict(),
                'gp_model_state_dict': gp_model.state_dict(),
                'likelihood_state_dict': likelihood.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss_min': early_stopping.val_loss_min,
                'best_score': early_stopping.best_score,
                'loss_history': loss_history # Save full history up to this point
            }, checkpoint_path)

        if early_stopping.early_stop:
            print(f"Early Stopping triggered at epoch {i+1}! No improvement for {patience} epochs.")
            break # Exit training loop

    print("Training finished.")

    # Load the best performing model from the saved checkpoint
    # This ensures the function returns the best state rather than the last one.
    if os.path.exists(checkpoint_path):
        print(f"Loading best joint model from {checkpoint_path} for final return...")
        best_checkpoint = torch.load(checkpoint_path, map_location=device)
        flow_model.load_state_dict(best_checkpoint['flow_model_state_dict'])
        gp_model.load_state_dict(best_checkpoint['gp_model_state_dict'])
        likelihood.load_state_dict(best_checkpoint['likelihood_state_dict'])
    else:
        print(f"Warning: No 'best model' checkpoint found at {checkpoint_path}. "
              "The returned model is the state from the last trained epoch.")

    # Set models to evaluation mode before returning
    flow_model.eval()
    gp_model.eval()
    likelihood.eval()

    return flow_model, gp_model, likelihood, loss_history