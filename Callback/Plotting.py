import matplotlib.pyplot as plt
import os

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

def plot_and_save_losses(train_losses, val_losses, save_dir='runs/train', filename='loss_graph.png'):
    """
    Plots training and validation losses and saves the graph as an image.
    
    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        save_dir (str): Directory where the image will be saved.
        filename (str): Name of the saved image file.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot the losses
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o', markersize=4)
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', marker='x', markersize=4)
    
    # Customize the graph
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_and_save_metrics(train_metrics, val_metrics, metric_name="Accuracy", save_dir='runs/train', filename='metric_graph.png'):
    """
    Plots training and validation metrics (e.g., accuracy) and saves the graph.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, label=f'Train {metric_name}', color='green', marker='o', markersize=4)
    plt.plot(epochs, val_metrics, label=f'Validation {metric_name}', color='orange', marker='x', markersize=4)
    
    plt.title(f'Training and Validation {metric_name}', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


class TensorBoardLogger:
    """Full TensorBoard logging: scalars, weight histograms, model graph, embeddings, and hparams."""

    def __init__(self, log_dir='runs/tensorboard'):
        if SummaryWriter is None:
            raise ModuleNotFoundError(
                "TensorBoard logging requires the 'tensorboard' package. "
                "Install it or leave tensorboard_log_dir=None."
            )
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, lr=None):
        self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        if lr is not None:
            self.writer.add_scalar('Learning_Rate', lr, epoch)

    def log_weight_histograms(self, model, epoch):
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'Weights/{name}', param, epoch)
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    def log_graph(self, model, input_tensor):
        self.writer.add_graph(model, input_tensor)

    def log_embeddings(self, embeddings, labels=None, epoch=None, tag='Embeddings'):
        metadata = labels.tolist() if labels is not None else None
        self.writer.add_embedding(embeddings, metadata=metadata, global_step=epoch, tag=tag)

    def log_hparams(self, hparam_dict, metric_dict):
        self.writer.add_hparams(hparam_dict, metric_dict)

    def close(self):
        self.writer.flush()
        self.writer.close()
