import matplotlib.pyplot as plt
import os

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

def plot_and_save_losses(train_losses, val_losses, save_dir='runs/train', filename='loss_graph.png'):
    """
    Plot training and validation losses and save the figure to disk.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        save_dir (str, optional): Directory where the image will be saved.
        filename (str, optional): Name of the saved image file.

    Returns:
        None
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
    Plot training and validation metrics and save the figure to disk.

    Args:
        train_metrics (list): Metric values measured on the training set.
        val_metrics (list): Metric values measured on the validation set.
        metric_name (str, optional): Human-readable metric name for plot labels.
        save_dir (str, optional): Directory where the image will be saved.
        filename (str, optional): Name of the saved image file.

    Returns:
        None
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
        """Initialize the TensorBoard summary writer.

        Args:
            log_dir (str, optional): Directory used to store TensorBoard event files.

        Returns:
            None
        """
        if SummaryWriter is None:
            raise ModuleNotFoundError(
                "TensorBoard logging requires the 'tensorboard' package. "
                "Install it or leave tensorboard_log_dir=None."
            )
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, lr=None):
        """Log per-epoch scalar metrics to TensorBoard.

        Args:
            epoch (int): Epoch index used as the TensorBoard global step.
            train_loss (float): Training loss for the epoch.
            val_loss (float): Validation loss for the epoch.
            train_acc (float): Training accuracy for the epoch.
            val_acc (float): Validation accuracy for the epoch.
            lr (float, optional): Learning rate to log for the epoch.

        Returns:
            None
        """
        self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        if lr is not None:
            self.writer.add_scalar('Learning_Rate', lr, epoch)

    def log_weight_histograms(self, model, epoch):
        """Log parameter and gradient histograms for a model.

        Args:
            model (torch.nn.Module): Model whose parameters should be logged.
            epoch (int): Epoch index used as the TensorBoard global step.

        Returns:
            None
        """
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'Weights/{name}', param, epoch)
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    def log_graph(self, model, input_tensor):
        """Log a model computation graph using a sample input tensor.

        Args:
            model (torch.nn.Module): Model to trace for graph visualization.
            input_tensor (torch.Tensor): Example input passed to the model.

        Returns:
            None
        """
        self.writer.add_graph(model, input_tensor)

    def log_embeddings(self, embeddings, labels=None, epoch=None, tag='Embeddings'):
        """Log embeddings and optional metadata for projector visualization.

        Args:
            embeddings (torch.Tensor): Embedding matrix to visualize.
            labels (torch.Tensor, optional): Labels converted to metadata strings.
            epoch (int, optional): Global step attached to the embedding snapshot.
            tag (str, optional): TensorBoard tag for the embedding entry.

        Returns:
            None
        """
        metadata = labels.tolist() if labels is not None else None
        self.writer.add_embedding(embeddings, metadata=metadata, global_step=epoch, tag=tag)

    def log_hparams(self, hparam_dict, metric_dict):
        """Log hyperparameters and summary metrics to TensorBoard.

        Args:
            hparam_dict (dict): Hyperparameter values to record.
            metric_dict (dict): Final metrics associated with the hyperparameters.

        Returns:
            None
        """
        self.writer.add_hparams(hparam_dict, metric_dict)

    def close(self):
        """Flush and close the TensorBoard writer.

        Args:
            None

        Returns:
            None
        """
        self.writer.flush()
        self.writer.close()
