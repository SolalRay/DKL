import torch.optim as optim
import numpy as np


class EarlyStopping:
    """
    Arrête l'entraînement tôt si la perte de validation ne s'améliore pas après une patience donnée.
    """
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf # Initialiser la perte minimale à l'infini)
        self.delta = delta

    def __call__(self, val_loss):
        """
        Appelée à chaque fin d'époque pour décider si l'entraînement doit s'arrêter.
        Retourne True si une amélioration est détectée, False sinon.
        """
        score = -val_loss
        
        improved = False

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss # Update min_loss
            improved = True
        elif score < self.best_score + self.delta: # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else: # Improvement detected
            self.best_score = score
            self.val_loss_min = val_loss # Update min_loss
            self.counter = 0 # Reset counter
            improved = True
        return improved # on donne le signal pour sauvegarder le modèle si amélioré
