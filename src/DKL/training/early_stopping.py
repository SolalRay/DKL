import torch.optim as optim
import numpy as np

class EarlyStopping:
    """
    Stops training early if the validation loss does not improve after a given patience.
    """
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf # Initialize minimum loss to infinity
        self.delta = delta

    def __call__(self, val_loss):
        """
        Called at the end of each epoch to decide whether training should stop.
        Returns True if an improvement is detected, False otherwise.
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
            
        return improved # Provides the signal to save the model if improved