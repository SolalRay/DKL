
import os
import torch
import gpytorch
import torch.optim as optim
from ..models.kernel import ExactGPModel
from ..models.kernel import TransformedGPModel
from ..training.early_stopping import EarlyStopping

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)



def train_naive_gp(common_data, realization, num_epochs=100, lr=0.1, 
                   checkpoint_name='naive_gp_checkpoint.pth', 
                   patience=10, 
                   delta=0):     
    """
    Entraîne le GP naïf avec Early Stopping.
    Sauvegarde le modèle seulement s'il y a amélioration et que l'époque est un multiple de 10.
    """
    checkpoint_path = MODELS_DIR / checkpoint_name
    device = common_data['X_train_torch'].device
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
    checkpoint_path = checkpoint_path # Le meilleur modèle sera sauvegardé ici

    if os.path.exists(checkpoint_path):
        print(f"Chargement du checkpoint Naive GP depuis {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        naive_gp_model.load_state_dict(checkpoint['gp_model_state_dict'])
        naive_likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        naive_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Reprise de l'entraînement Naive GP à partir de l'époque {start_epoch}")
    else:
        print("Aucun checkpoint Naive GP trouvé. Début de l'entraînement.")

    naive_gp_model.train()
    naive_likelihood.train()

    print(f"Entraînement du GP Naif sur les données originales pour {num_epochs} époques (début {start_epoch})...")
    for k in range(start_epoch, num_epochs):
        naive_optimizer.zero_grad()
        output_naive = naive_gp_model(X_train_torch)
        loss_naive = -mll_naive(output_naive, Y_train_torch)
        loss_naive.backward()
        naive_optimizer.step()

        current_loss_item = loss_naive.item()

        # Logique d'Early Stopping et d'enregistrement
        improved = early_stopping(current_loss_item)

        if improved and (k + 1) % 10 == 0: 
            print(f'  Naive GP Époque {k+1}/{num_epochs}: Perte {current_loss_item:.4f}, Longueur de corrélation: {naive_gp_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()}, Échelle de sortie: {naive_gp_model.covar_module.outputscale.item():.4f}, Bruit: {naive_likelihood.noise.item():.4f}')
            print(f"  Meilleure perte détectée et époque éligible.")
            torch.save({
                'epoch': k,
                'gp_model_state_dict': naive_gp_model.state_dict(),
                'likelihood_state_dict': naive_likelihood.state_dict(),
                'optimizer_state_dict': naive_optimizer.state_dict(),
                'val_loss_min': early_stopping.val_loss_min,
                'best_score': early_stopping.best_score,
            }, checkpoint_path)

        if early_stopping.early_stop:
            print(f"Early Stopping activé à l'époque {k+1}! Aucune amélioration depuis {patience} époques.")
            break 

    print("Entraînement Naive GP terminé.")

    # Chargement du MEILLEUR modèle sauvegardé par Early Stopping
    if os.path.exists(checkpoint_path):
        print(f"Chargement du meilleur modèle Naive GP depuis {checkpoint_path} pour le retour final...")
        best_checkpoint = torch.load(checkpoint_path, map_location=device)
        naive_gp_model.load_state_dict(best_checkpoint['gp_model_state_dict'])
        naive_likelihood.load_state_dict(best_checkpoint['likelihood_state_dict'])
    else:
        print(f"Attention : Aucun checkpoint du meilleur modèle trouvé à {checkpoint_path}. "
              "Le modèle retourné est celui de la dernière époque entraînée.")

    naive_gp_model.eval()
    naive_likelihood.eval()

    return naive_gp_model, naive_likelihood


def train_ideal_gp(common_data, realization,function,
                num_epochs=100, lr=0.1,
                    lengthscale = 2.0,
                   checkpoint_name='ideal_gp_checkpoint.pth', 
                   patience=10, 
                   delta=0):     
    """
    Entraîne le GP idéal sur des données transformées avec Early Stopping.
    Sauvegarde le modèle seulement s'il y a amélioration et que l'époque est un multiple de 10.
    """
    device = common_data['X_train_torch'].device
    X_train_torch = common_data['X_train_torch'].to(device)
    Y_train_torch = realization['Y_train_torch'].to(device)
    checkpoint_path = MODELS_DIR / checkpoint_name


    ideal_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    ideal_gp_model = TransformedGPModel(X_train_torch, Y_train_torch, ideal_likelihood, function, lengthscale).to(device) # Assurez-vous que TransformedGPModel est défini

    ideal_optimizer = optim.Adam([
        {'params': ideal_gp_model.parameters()},
    ], lr=lr)

    mll_ideal = gpytorch.mlls.ExactMarginalLogLikelihood(ideal_likelihood, ideal_gp_model)

    start_epoch = 0
    early_stopping = EarlyStopping(patience=patience, delta=delta)
    checkpoint_path = checkpoint_path # Le meilleur modèle sera sauvegardé ici

    if os.path.exists(checkpoint_path):
        print(f"Chargement du checkpoint GP transformé depuis {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ideal_gp_model.load_state_dict(checkpoint['gp_model_state_dict'])
        ideal_likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        ideal_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Reprise de l'entraînement GP transformé à partir de l'époque {start_epoch}")
    else:
        print("Aucun checkpoint GP transformé trouvé. Début de l'entraînement.")

    ideal_gp_model.train()
    ideal_likelihood.train()

    print(f"Entraînement du GP transformé pour {num_epochs} époques (début {start_epoch})...")
    for k in range(start_epoch, num_epochs):
        ideal_optimizer.zero_grad()
        output_ideal = ideal_gp_model(X_train_torch)
        loss_ideal = -mll_ideal(output_ideal, Y_train_torch)
        loss_ideal.backward()
        ideal_optimizer.step()

        current_loss_item = loss_ideal.item()

        # Logique d'Early Stopping et d'enregistrement
        improved = early_stopping(current_loss_item)

        if improved and (k + 1) % 10 == 0: 
            print(f'  GP transformé Époque {k+1}/{num_epochs}: Perte {current_loss_item:.4f}, Longueur de corrélation: {ideal_gp_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()}, Échelle de sortie: {ideal_gp_model.covar_module.outputscale.item():.4f}, Bruit: {ideal_likelihood.noise.item():.4f}')
            print(f"  Meilleure perte détectée et époque éligible.")
            torch.save({
                'epoch': k,
                'gp_model_state_dict': ideal_gp_model.state_dict(),
                'likelihood_state_dict': ideal_likelihood.state_dict(),
                'optimizer_state_dict': ideal_optimizer.state_dict(),
                'val_loss_min': early_stopping.val_loss_min,
                'best_score': early_stopping.best_score,
            }, checkpoint_path)

        if early_stopping.early_stop:
            print(f"Early Stopping activé à l'époque {k+1}! Aucune amélioration depuis {patience} époques.")
            break 

    print("Entraînement GP transformé terminé.")

    # Chargement du MEILLEUR modèle sauvegardé par Early Stopping
    if os.path.exists(checkpoint_path):
        print(f"Chargement du meilleur modèle Ideal GP depuis {checkpoint_path} pour le retour final...")
        best_checkpoint = torch.load(checkpoint_path, map_location=device)
        ideal_gp_model.load_state_dict(best_checkpoint['gp_model_state_dict'])
        ideal_likelihood.load_state_dict(best_checkpoint['likelihood_state_dict'])
    else:
        print(f"Attention : Aucun checkpoint du meilleur modèle trouvé à {checkpoint_path}. "
              "Le modèle retourné est celui de la dernière époque entraînée.")

    ideal_gp_model.eval()
    ideal_likelihood.eval()

    return ideal_gp_model, ideal_likelihood
