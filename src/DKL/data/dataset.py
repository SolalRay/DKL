import numpy as np
import torch
import gstlearn as gl
import numpy as np
import numpy as np

def modelgpytorch(nu,scale):
    """Creates a Matern covariance model using gstlearn compatible with gpytorch's Matern kernel."""
    return gl.Model.createFromParam(gl.ECov.MATERN,param = nu, range = 1./np.sqrt(2 * nu) * scale, flagRange = False)

def generation_approchee(x,y,nu,scale,noise):
    db = gl.Db.create()
    db["x"] = x
    db["y"] = y
    db.setLocators(["x","y"],gl.ELoc.X)
    model =  modelgpytorch(nu,scale)
    gl.simtub(None,db,model, nbtuba = 10000)
    return db["Simu"] 


def generate_2d_data(transformation, n_train_points=1000, test_grid_size=100, scale=5.,
                       lengthscale=2.0, nu=2.5, noise_level=0.05,
                       base_seed=None, device='cpu', ):
    """
    Génère une realisation d'un processus gaussien sur des points aléatoires pour l'entrainement
    et sur un maillage régulier pour le test. Le GP est généré sur l'union des points d'entraînement et de test. 
    Attention, la génération GP se fait sur l'espace transformé defini par la la fonction definie dans le module.
    """
    print(f"  - {n_train_points} points aléatoires d'entraînement par realisation")
    print(f"  - Maillage de visualisation {test_grid_size}x{test_grid_size} ({test_grid_size*test_grid_size} points) pour le test")

    if base_seed is not None:
        master_rng = np.random.RandomState(base_seed)
    else:
        master_rng = np.random.RandomState()

    # Generation des points d'entrainement
    X_train_np = master_rng.uniform(-scale, scale, (n_train_points, 2))  # points 2D numpy
    X_train_torch = torch.tensor(X_train_np, dtype=torch.float32).to(device)   # conversion torch

    # Generation des points de test
    x_grid = np.linspace(-scale, scale, test_grid_size)
    y_grid = np.linspace(-scale, scale, test_grid_size)
    
    X_test_grid_grid_np, Y_test_grid_grid_np = np.meshgrid(x_grid, y_grid)
    X_test_grid_np = np.vstack([X_test_grid_grid_np.ravel(), Y_test_grid_grid_np.ravel()]).T
    X_test_grid_torch = torch.tensor(X_test_grid_np, dtype=torch.float32).to(device)

    # Stacking des points pour la generation du GP, puis transformation par la fonction definie dans le module fonction.py
    X_combined_np = np.vstack([X_train_np, X_test_grid_np])
    X_combined_transformed_np_for_gen = transformation(X_combined_np, scale=lengthscale)

    # Stocker les données communes (points et paramètres de grille/scale)
    common_data = {
        'X_train_np': X_train_np, # Points d'entraînement NumPy
        'X_train_torch': X_train_torch, # Points d'entraînement Torch
        'X_test_grid_np': X_test_grid_np, # Points de grille aplatis NumPy
        'X_test_grid_torch': X_test_grid_torch, # Points de grille aplatis Torch
        'X_test_grid_np': X_test_grid_grid_np, # Points de grille X pour meshgrid NumPy
        'Y_test_grid_np': Y_test_grid_grid_np, # Points de grille Y pour meshgrid NumPy
        'test_grid_size': test_grid_size,
        'n_train_points': n_train_points,
        'scale': scale,
    }
    # Generation par la bibliotheque de gstlearn d'une realisation du processus gaussien sur les points d'entrainement et de test
    y_combined_np = generation_approchee(X_combined_transformed_np_for_gen[:,0],
                                         X_combined_transformed_np_for_gen[:,1],
                                         nu ,scale,noise_level)

    # On separe les points de donnees d'entrainement et de test
    y_train_np = y_combined_np[:n_train_points] +  np.sqrt(noise_level) * np.random.normal(size = n_train_points) # On ajoute du bruit aux points d'entrainement
    # On separe les points de test
    y_test_grid_np = y_combined_np[n_train_points:] # Le reste est pour la grille de test
    y_train_torch = torch.tensor(y_train_np, dtype=torch.float32).to(device)
    y_test_grid_torch = torch.tensor(y_test_grid_np, dtype=torch.float32).to(device)

    # Remodeler les valeurs de grille en 2D pour une visualisation potentielle directe
    y_test_grid_2d = y_test_grid_np.reshape(test_grid_size, test_grid_size)

    # Stocker les données de cette réalisation
    realization = {
        'Y_train_np': y_train_np, # y aux points d'entraînement format np
        'Y_train_torch': y_train_torch, # y aux points d'entraînement format torch
        'Y_test_grid_np': y_test_grid_np, # y aux points de test grille np aplati
        'Y_test_grid_torch': y_test_grid_torch, # y aux points de test grille torch aplati
        'Y_test_grid_2d': y_test_grid_2d, # y aux points de test grille np grille 2D
    }
    print(f"Stats Y (train) min={y_train_np.min():.4f}, max={y_train_np.max():.4f}, mean={y_train_np.mean():.4f}, std={y_train_np.std():.4f}")
    print(f"Stats Y (test grid) min={y_test_grid_np.min():.4f}, max={y_test_grid_np.max():.4f}, mean={y_test_grid_np.mean():.4f}, std={y_test_grid_np.std():.4f}")


    print("Génération de données terminée.")

    return common_data, realization
