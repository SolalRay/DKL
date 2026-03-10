import os
import torch
import gpytorch
import torch.optim as optim
from tqdm import tqdm
from ..models.kernel import ExactGPModel
from ..models.kernel import TransformedGPModel
from ..training.early_stopping import EarlyStopping
from ..models.normalizing_flow import RealNVP


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

def train_hybrid_flow_gp_simple(common_data, realization, num_epochs=500, flow_lr=0.001, gp_lr=0.01, 
                               flow_batch_size=32, gp_update_frequency=10, no_learn_lengthscale=False,
                               checkpoint_name='hybrid_checkpoint.pth', patience=10, delta=0,
                               device = "cpu", num_flow_blocks=12):
    """
    Version simplifiée qui évite complètement le problème d'indexation.
    Utilise un GP temporaire pour chaque minibatch.
    """
    
    flow_model = RealNVP(num_blocks=num_flow_blocks).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    checkpoint_path = MODELS_DIR / checkpoint_name

    
    # GP principal initialisé avec toutes les données
    X_train_torch = common_data['X_train_torch'].to(device)
    Y_train_torch = realization['Y_train_torch'].to(device)  # Première réalisation pour init
    gp_model = ExactGPModel(X_train_torch, Y_train_torch, likelihood).to(device)
    
    if no_learn_lengthscale:
        gp_model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
        gp_model.covar_module.base_kernel.lengthscale = 1.0
    
    # Optimiseurs séparés
    flow_optimizer = optim.Adam(flow_model.parameters(), lr=flow_lr)
    gp_params = list(filter(lambda p: p.requires_grad, gp_model.parameters()))
    gp_optimizer = optim.Adam(gp_params, lr=gp_lr) if gp_params else None
    
    # Préparer les minibatches pour le Flow
    num_samples = X_train_torch.shape[0]
    num_flow_batches = (num_samples + flow_batch_size - 1) // flow_batch_size
    
    print(f"Stratégie hybride simple:")
    print(f"- Flow: minibatch de {flow_batch_size} échantillons")
    print(f"- GP: toutes les données ({num_samples} échantillons)")
    print(f"- Mise à jour GP: tous les {gp_update_frequency} steps")
    
    flow_model.train()
    gp_model.train()
    likelihood.train()
    
    step_count = 0
    loss_history = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Boucle sur les réalisations
        Y_real = realization['Y_train_torch'].to(device)
        
        # Entraîner le Flow par minibatch
        for batch_idx in range(num_flow_batches):
            step_count += 1
            update_gp = (step_count % gp_update_frequency == 0)
            
            # Préparer le minibatch pour le Flow
            start_idx = batch_idx * flow_batch_size
            end_idx = min((batch_idx + 1) * flow_batch_size, num_samples)
            
            X_batch = X_train_torch[start_idx:end_idx]
            Y_batch = Y_real[start_idx:end_idx]
            
            # Forward pass Flow
            flow_optimizer.zero_grad()
            if update_gp and gp_optimizer:
                gp_optimizer.zero_grad()
            
            # Transformer le minibatch
            transformed_batch = flow_model(X_batch)
            
            if update_gp:
                # Créer un GP temporaire pour ce minibatch
                temp_gp = ExactGPModel(transformed_batch, Y_batch, likelihood).to(device)
                
                # Copier les paramètres du GP principal
                temp_gp.load_state_dict(gp_model.state_dict())
                
                # Évaluer
                output_dist = temp_gp(transformed_batch)
                temp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, temp_gp)
                loss = -temp_mll(output_dist, Y_batch)
                
                # Mettre à jour le GP principal après le backward
                loss.backward()
                
                # Copier les gradients vers le GP principal
                for main_param, temp_param in zip(gp_model.parameters(), temp_gp.parameters()):
                    if main_param.requires_grad and temp_param.grad is not None:
                        if main_param.grad is None:
                            main_param.grad = temp_param.grad.clone()
                        else:
                            main_param.grad += temp_param.grad
            else:
                # Seulement le Flow, GP fixé
                with torch.no_grad():
                    # Utiliser le GP principal mais avec des données transformées
                    temp_gp = ExactGPModel(transformed_batch, Y_batch, likelihood).to(device)
                    temp_gp.load_state_dict(gp_model.state_dict())
                    
                    # Évaluer
                    output_dist = temp_gp(transformed_batch)
                    temp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, temp_gp)
                
                # Recalculer avec gradients pour le Flow
                loss = -temp_mll(output_dist, Y_batch)
                loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
            if update_gp and gp_optimizer:
                torch.nn.utils.clip_grad_norm_(gp_params, max_norm=1.0)
            
            # Mise à jour
            flow_optimizer.step()
            if update_gp and gp_optimizer:
                gp_optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(realization) * num_flow_batches)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Époque {epoch+1}: Loss = {avg_loss:.6f}")
    
    return flow_model, gp_model, likelihood, loss_history
