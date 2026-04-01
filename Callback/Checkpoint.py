import os
import torch

def save_checkpoint(state, is_best, save_dir, last_name='last.pt', best_name='best.pt'):
    """
    Saves the training state to disk.
    
    Args:
        state (dict): A dictionary containing 'epoch', 'state_dict', 'optimizer', and potentially others.
        is_best (bool): If True, also saves this state as the best model.
        save_dir (str): Directory to save the checkpoints.
        last_name (str): Filename for the latest checkpoint.
        best_name (str): Filename for the best checkpoint.
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
    Loads the training state from disk to resume training.
    
    Args:
        checkpoint_path (str): Path to the saved checkpoint (e.g., 'runs/train/last.pt').
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler, optional): The scheduler to load state into.
        
    Returns:
        tuple: (model, optimizer, lr_scheduler, start_epoch, best_loss)
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
    Saves only the backbone weights (without the head) for ReID inference.
    
    Args:
        backbone (torch.nn.Module): The backbone module to save.
        save_dir (str): Directory to save the weights.
        filename (str): Filename for the backbone weights.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(backbone.state_dict(), path)

def load_backbone_weights(backbone, checkpoint_path):
    """
    Loads backbone-only weights from disk.
    
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
