import os
import torch

def save_checkpoint(state, is_best, save_dir, last_name='last.pt', best_name='best.pt'):
    """
    Save a training checkpoint to disk and optionally mark it as the best state.

    Args:
        state (dict): Dictionary containing the serialized training state.
        is_best (bool): If True, also saves this state as the best model.
        save_dir (str): Directory to save the checkpoints.
        last_name (str, optional): Filename for the latest checkpoint.
        best_name (str, optional): Filename for the best checkpoint.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Always save the last state
    last_path = os.path.join(save_dir, last_name)
    torch.save(state, last_path)
    
    # Save the best state if needed
    if is_best:
        best_path = os.path.join(save_dir, best_name)
        torch.save(state, best_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, lr_scheduler=None):
    """
    Load a training checkpoint into the provided model and optimizer objects.

    Args:
        checkpoint_path (str): Path to the saved checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): The scheduler to load state into.

    Returns:
        tuple: Tuple containing the model, optimizer, scheduler, next epoch index,
        and best validation loss recovered from the checkpoint.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
        
    checkpoint = torch.load(checkpoint_path)
    
    # Load model weights
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else: # Fallback in case raw model weights were saved
        model.load_state_dict(checkpoint)
    
    # Load optimizer state
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    # Load scheduler state
    if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
    start_epoch = checkpoint.get('epoch', 0) + 1 # Start from the next epoch
    best_loss = checkpoint.get('best_loss', float('inf'))
    
    return model, optimizer, lr_scheduler, start_epoch, best_loss


def save_backbone_weights(backbone, save_dir, filename='best_backbone.pt'):
    """
    Save only the backbone weights for inference or reuse.

    Args:
        backbone (torch.nn.Module): The backbone module to save.
        save_dir (str): Directory to save the weights.
        filename (str, optional): Filename for the backbone weights.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(backbone.state_dict(), path)

def load_backbone_weights(backbone, checkpoint_path):
    """
    Load backbone-only weights from disk into the provided module.

    Args:
        backbone (torch.nn.Module): The backbone module to load weights into.
        checkpoint_path (str): Path to the saved backbone weights.

    Returns:
        torch.nn.Module: The backbone with loaded weights.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No backbone weights found at '{checkpoint_path}'")
    backbone.load_state_dict(torch.load(checkpoint_path))
    return backbone
