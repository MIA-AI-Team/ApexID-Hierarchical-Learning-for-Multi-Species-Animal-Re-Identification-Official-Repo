import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0.0, path='checkpoint.pt'):
        """
        Initialize the early stopping monitor.

        Args:
            patience (int, optional): Number of non-improving evaluations to tolerate
                before stopping.
            verbose (bool, optional): Whether to print progress messages when the
                validation loss changes.
            delta (float, optional): Minimum improvement required to reset the
                patience counter.
            path (str, optional): Destination path used to save the best model state.

        Returns:
            None
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        """Update the early stopping state using a new validation loss value.

        Args:
            val_loss (float): Current validation loss to compare against the best
                recorded score.
            model (torch.nn.Module): Model whose parameters are saved when the loss
                improves.

        Returns:
            bool: True when the current validation loss is a new best score,
            otherwise False.
        """
        score = -val_loss
        is_best = False

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            is_best = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            is_best = True
        
        return is_best

    def save_checkpoint(self, val_loss, model):
        """Save the model parameters when validation loss improves.

        Args:
            val_loss (float): Validation loss value being recorded as the new best.
            model (torch.nn.Module): Model whose parameters should be saved.

        Returns:
            None
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
