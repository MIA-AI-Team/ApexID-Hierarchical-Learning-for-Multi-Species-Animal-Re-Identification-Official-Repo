import math
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
import pandas as pd
import random
from torch.utils.data import DataLoader
from Dataset.Dataset import ApexID_Dataset

class LRFinder:
    def __init__(self, backbone, head, optimizer, train_ds_path, train_transform, train_label_to_idx,
                 metrics_label_map, batch_size, rootdir, device=None, save_path = "LRFinder_history.csv"):
        self.backbone = backbone
        self.head = head
        self.optimizer = optimizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        self.train_ds_path = train_ds_path
        self.train_transform = train_transform
        self.train_label_to_idx = train_label_to_idx
        self.metrics_label_map = metrics_label_map
        self.batch_size = batch_size
        self.rootdir = rootdir
        self.open_set_ignore = -100
        self.backbone.to(self.device)
        self.head.to(self.device)
        
        self.history = {"log_lr": [], "smoothed_loss": []}
        self.best_loss = float('inf')

    def run(self,  init_lr=1e-7, final_lr=10.0, beta=0.985):
        """Runs Leslie Smith's Learning Rate Range Test."""
        
        train_loader = self._data_split()
        num_batches = len(train_loader) - 1
        if num_batches <= 0:
            raise ValueError("Train loader is too small for the LR finder.")
        
        lr_mult = (final_lr / init_lr) ** (1 / num_batches)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = init_lr
        lr = init_lr

        avg_loss = 0.0
        self.best_loss = float('inf')
        batch_num = 0

        self.history = {"log_lr": [], "smoothed_loss": []}

        self.backbone.train()
        self.head.train()

        print("Starting Leslie Smith's LR Range Test")

        for images, cls_labels, _ in tqdm(train_loader, desc="LR Finder"):
            batch_num += 1
            
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            embeddings = self.backbone(images)
            loss = self.head(embeddings, cls_labels) 
            
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            
            if batch_num == 1 or smoothed_loss < self.best_loss:
                self.best_loss = smoothed_loss
                
            if batch_num > 1 and smoothed_loss > 4 * self.best_loss:
                print("Loss diverging, stopping early to save time.")
                break
            
            self.history["smoothed_loss"].append(smoothed_loss)
            self.history["log_lr"].append(math.log10(lr))
            
            loss.backward()
            self.optimizer.step()
            
            # Update the learning rate for the next batch
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.save_path, index=False)
                
        return self.history
    
    def _data_split(self):
        label_col = "Identity"
        df = pd.read_csv(self.train_ds_path)
        
        unique_ids = df[label_col].unique()
        num_sampled_ids = max(1, int(len(unique_ids) * 0.10))
        sampled_ids = random.sample(list(unique_ids), num_sampled_ids)
        
        sampled_df = df[df[label_col].isin(sampled_ids)].reset_index(drop=True)
        print(f"Sampled {len(sampled_df)} images across {len(sampled_ids)} identities.")
        train_dataset = ApexID_Dataset(
                            sampled_df,
                            transform=self.train_transform,
                            label_to_idx=self.train_label_to_idx,      
                            root_dir=self.rootdir,
                            allow_unknown_identity=False,
                            unknown_label=self.open_set_ignore,
                            metrics_label_map=self.metrics_label_map,   
                        )
        
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def plot(self, save_plot_path = "/LR_plot.png"):
        """
        In this plot we look for the lowest range of learning rates where the loss is still decreasing. 
        We want to avoid the region where the loss starts to diverge, which typically happens at higher learning rates. 
        The ideal learning rate is often found just before the point where the loss begins to increase sharply.
        """
        if not self.history["log_lr"]:
            print("No data to plot. Please call run() first.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.history["log_lr"], self.history["smoothed_loss"], lw=2)

        ticks = plt.xticks()[0]
        labels = [f"1e{int(t)}" for t in ticks]
        plt.xticks(ticks, labels)

        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Smoothed Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True, linestyle='--', alpha=0.7)
        if save_plot_path:
            plt.savefig(save_plot_path)
        plt.show()