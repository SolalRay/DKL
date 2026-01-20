import numpy as np

def transform_tanh(x, scale=5.):
    scaling_factor = scale  # Pour rester dans l'intervalle [-5,5]
    result = scaling_factor * np.tanh(x)
    return result

def transform_x_abs_x(x,scale=5.):
    """Applique la transformation x*|x| avec un centre donné"""
    offset = x
    r = np.linalg.norm(offset, axis=1, keepdims=True)
    result = offset*r
    return result

def transform_sin(points, scale=5.0, amp_x=1, freq_x=1, amp_y=2, freq_y=1):
    """
    Distorsion par vagues sinusoïdales - compatible PyTorch et NumPy
    """
    # Vérifier si l'entrée est un tensor PyTorch ou un array NumPy
    is_torch = hasattr(points, 'dtype') and 'torch' in str(points.dtype)
    
    if is_torch:
        import torch
        # Version PyTorch
        x, y = points[:, 0], points[:, 1]
        x_new = x + amp_x * torch.sin(freq_x * y)
        y_new = y + amp_y * torch.sin(freq_y * x)
        return torch.stack((x_new, y_new), dim=1)
    else:
        import numpy as np
        # Version NumPy
        x, y = points[:, 0], points[:, 1]
        x_new = x + amp_x * np.sin(freq_x * y)
        y_new = y + amp_y * np.sin(freq_y * x)
        return np.column_stack((x_new, y_new))

def transform_spiral(x, centre=np.array([0., 0.]), num_twists=2):
    """Transforme les points en appliquant une distorsion en spirale."""
    offset = x - centre  # Décalage par rapport au centre
    r = np.linalg.norm(offset, axis=1)#, keepdims=True)  # Distance au centre (shape: (num_samples,))
    theta = np.arctan2(offset[:, 1], offset[:, 0])  # Angle initial

    # Ajouter une rotation proportionnelle à r pour créer une spirale
    theta_new = theta + num_twists * np.pi * r/4  # Spiral effect

    # Nouveaux points transformés
    result = np.stack([r * np.cos(theta_new), r * np.sin(theta_new)], axis=1) + centre
    return result


def injective_multi_spiral(z, num_centers=3, twist_strength=2.0, influence_radius=0.8):
    """
    Transformation injective continue avec plusieurs tourbillons.
    
    Paramètres :
    - z : (N, 2) échantillons uniformes ∈ [-1, 1]^2
    - num_centers : nombre de centres de rotation
    - twist_strength : force de la rotation
    - influence_radius : rayon d’influence d’un tourbillon

    Retourne :
    - x : (N, 2) points transformés
    """
    x = z.copy()

    # Générer des centres répartis en cercle
    angles = np.linspace(0, 2 * np.pi, num_centers, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 0.8  # placés dans [-1, 1]^2

    for center in centers:
        offset = x - center  # vecteurs centre → points
        dist = np.linalg.norm(offset, axis=1, keepdims=True)

        # Fonction d’atténuation radiale (évite interactions multiples)
        mask = np.exp(-(dist**2) / (2 * influence_radius**2))  # ∈ [0,1]

        # Rotation orthogonale du vecteur offset (90° dans le plan)
        rotated = np.stack([-offset[:, 1], offset[:, 0]], axis=1)

        # Ajout du champ de rotation pondéré
        x += twist_strength * mask * rotated

    return x
