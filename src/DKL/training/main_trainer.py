import os
import torch
import torch.optim as optim
import gpytorch
from DKL.models.normalizing_flow import RealNVP
from DKL.models.kernel import ExactGPModel
from DKL.training.early_stopping import EarlyStopping

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_joint_model(common_data, realization, num_epochs=500, flow_lr=0.001, gp_lr=0.01, num_flow_blocks=12, no_learn_lengthscale=False,
                      checkpoint_name='learned_gp_checkpoint.pth', # Chemin pour les checkpoints périodiques
                      patience=10, # patience pour Early Stopping
                      delta=0     #  delta pour Early Stopping
                      ): # Nouveau chemin pour le meilleur modèle par ES
    """
    Entraine le flow et le GP en meme temps, avec Early Stopping.
    Taux d'apprentissage differencie. Possibilite de ne pas apprendre la lengthscale du GP.
    Permet de reprendre l'entraînement à partir d'un checkpoint.
    """

    device = common_data['X_train_torch'].device
    flow_model = RealNVP(num_blocks=num_flow_blocks).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    checkpoint_path = MODELS_DIR / checkpoint_name

    # Initialize GP model with dummy data
    dummy_train_x = torch.zeros_like(common_data['X_train_torch']).to(device)
    dummy_train_y = torch.zeros_like(realization['Y_train_torch']).to(device)
    gp_model = ExactGPModel(dummy_train_x, dummy_train_y, likelihood).to(device)

    # Freezing the Lengthscale
    if no_learn_lengthscale:
        print("\nGeler la lengthscale...")
        gp_model.covar_module.base_kernel.raw_lengthscale.requires_grad = False

    # Separe les differents parametres
    param_groups = [
    {'params': flow_model.parameters(), 'lr': flow_lr},
    ]
    if not no_learn_lengthscale:
    # ajoute les parametres entrainables du GP si ils sont disponibles
        gp_learnable_params = list(filter(lambda p: p.requires_grad, gp_model.parameters()))
    if gp_learnable_params:
        param_groups.append({'params': gp_learnable_params, 'lr': gp_lr})

    optimizer = optim.Adam(param_groups)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    start_epoch = 0
    loss_history = []
    
    # --- Initialisation de l'EarlyStopping ---
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    # Si on reprend l'entrainement a un point donne
    if os.path.exists(checkpoint_path):
        print(f"Chargement du checkpoint depuis {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        flow_model.load_state_dict(checkpoint['flow_model_state_dict'])
        gp_model.load_state_dict(checkpoint['gp_model_state_dict'])
        likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss_history = checkpoint['loss_history']
        print(f"Reprise de l'entraînement à partir de l'époque {start_epoch}")
    else:
        print("Aucun checkpoint trouvé. Début de l'entraînement à zéro.")

    flow_model.train()
    gp_model.train()
    likelihood.train()

    print("\nDébut de l'entraînement...")

    for i in range(start_epoch, num_epochs): # Boucle depuis start_epoch
        optimizer.zero_grad()

        Y_train_torch = realization['Y_train_torch'].to(device) # Assurez-vous que les données sont sur le device
        X_train_torch = common_data['X_train_torch'].to(device) # Assurez-vous que les données sont sur le device

        # transforme les coordonnees par le NF
        transformed_z = flow_model(X_train_torch)

        # Entraine le GP sur la sortie du NF
        gp_model.set_train_data(transformed_z.detach(), Y_train_torch, strict=False)
        output_distribution = gp_model(transformed_z)

        # calcule la perte
        loss = -mll(output_distribution, Y_train_torch)


        # Effectue le backward pass sur la perte totale de l'époque
        loss.backward()

        ## Gradient clipping en cas de freezing de la lengthscale
        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
        if not no_learn_lengthscale:
             gp_learnable_params_now = list(filter(lambda p: p.requires_grad and p.grad is not None, gp_model.parameters()))
             if gp_learnable_params_now:
                  torch.nn.utils.clip_grad_norm_(gp_learnable_params_now, max_norm=1.0)

        optimizer.step()
        loss_history.append(loss.item())

        current_total_loss = loss.item()

        improved = early_stopping(current_total_loss)

        if improved and (i + 1) % 10:
            torch.save({
                'epoch': i,
                'flow_model_state_dict': flow_model.state_dict(),
                'gp_model_state_dict': gp_model.state_dict(),
                'likelihood_state_dict': likelihood.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss_min': early_stopping.val_loss_min,
                'best_score': early_stopping.best_score,
                'loss_history': loss_history # Sauvegarder l'historique complet jusqu'à ce point
            }, checkpoint_path)

        if early_stopping.early_stop:
            print(f"Early Stopping activé à l'époque {i+1} ! Aucune amélioration depuis {patience} époques.")
            break # Sort de la boucle d'entraînement

    print("Entraînement terminé.")

    # On change les le dernier modele periodique par le meilleur
    # Ceci garantit que la fonction retourne le modèle le plus performant.
    if os.path.exists(checkpoint_path):
        print(f"Chargement du meilleur modèle joint depuis {checkpoint_path} pour le retour final...")
        best_checkpoint = torch.load(checkpoint_path, map_location=device)
        flow_model.load_state_dict(best_checkpoint['flow_model_state_dict'])
        gp_model.load_state_dict(best_checkpoint['gp_model_state_dict'])
        likelihood.load_state_dict(best_checkpoint['likelihood_state_dict'])
    else:
        print(f"Attention : Aucun checkpoint du meilleur modèle trouvé à {checkpoint_path}. "
              "Le modèle retourné est celui de la dernière époque entraînée.")

    # Passer les modèles en mode évaluation avant de les retourner
    flow_model.eval()
    gp_model.eval()
    likelihood.eval()

    return flow_model, gp_model, likelihood, loss_history
