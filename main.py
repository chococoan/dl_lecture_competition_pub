import os
import numpy as np
import pandas as pd
import io
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from scipy import signal
import sys
from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

def preprocess_single_sample(data, original_fs, target_fs, 
                             resample=True, scale=True):
    if resample:
        data = resample_data(data, original_fs, target_fs)
    if scale:
        data = scale_data(data)
    return data

def resample_data(data, original_fs, target_fs):
    num_samples = int(data.shape[1] * target_fs / original_fs) 
    data_resampled = signal.resample(data, num_samples, axis=1)
    return data_resampled

def scale_data(data):
    scaler = StandardScaler()
    # スケーリング前のデータの形状を維持
    data_reshaped = data.reshape(-1, data.shape[-1])
    data_scaled = scaler.fit_transform(data_reshaped).reshape(data.shape)
    return data_scaled

class ThingsMEGDatasetOnTheFly(ThingsMEGDataset):
    def __init__(self, split: str, data_dir: str = "data", original_fs=200, target_fs=100, 
                 resample=True, scale=True, time_shift=0) -> None:
        super().__init__(split, data_dir, time_shift)
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.resample = resample
        self.scale = scale

    def __getitem__(self, i):
        data = self.X[i]
        data = preprocess_single_sample(data.numpy(), self.original_fs, self.target_fs, 
                                        self.resample, self.scale)
        data = torch.tensor(data, dtype=torch.float32)
        subject_idx = self.subject_idxs[i] 
        if self.split == 'test':
            return data, subject_idx
        else:
            label = self.y[i]
            return data, label, subject_idx

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
  
        set_seed(1234)
        logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        
        if args.use_wandb:
            wandb.init(mode="online", dir=logdir, project="MEG-classification")

        print("Loading MEG datasets...")
        train_set = ThingsMEGDatasetOnTheFly("train", args.data_dir, resample=True, scale=False, time_shift=args.time_shift)
        val_set = ThingsMEGDatasetOnTheFly("val", args.data_dir, resample=True, scale=False)
        test_set = ThingsMEGDatasetOnTheFly("test", args.data_dir, resample=True, scale=False)

        loader_args = {"batch_size": 64, "num_workers": args.num_workers}
        
        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
        test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=64, num_workers=args.num_workers)

        print("Model initializing")
        model = BasicConvClassifier(
            train_set.num_classes, train_set.seq_len, train_set.num_channels, num_subjects=len(set(train_set.subject_idxs))
        ).to(args.device)
        print("Model initialized")

        print("Optimizer initializing")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        print("Optimizer initialized")

        max_val_acc = 0
        accuracy = Accuracy(
            task="multiclass", num_classes=train_set.num_classes, top_k=10
        ).to(args.device)

        early_stop_counter = 0
        early_stop_patience = 10

        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")

            train_loss, train_acc, val_loss, val_acc = [], [], [], []

            model.train()
            for X, y, subject_idx in tqdm(train_loader, desc="Train"):
                X = X.float().to(args.device)
                y = y.to(args.device)
                subject_idx = subject_idx.to(args.device)

                y_pred = model(X, subject_idx)  

                loss = F.cross_entropy(y_pred, y)
                train_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = accuracy(y_pred, y)
                train_acc.append(acc.item())

            model.eval()
            for X, y, subject_idx in tqdm(val_loader, desc="Validation"):
                X = X.to(args.device)
                y = y.to(args.device)
                subject_idx = subject_idx.to(args.device)

                with torch.no_grad():
                    y_pred = model(X, subject_idx)  

                val_loss.append(F.cross_entropy(y_pred, y).item())
                val_acc.append(accuracy(y_pred, y).item())

            print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
            torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
            if args.use_wandb:
                wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})

            if np.mean(val_acc) > max_val_acc:
                cprint("New best.", "cyan")
                torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
                max_val_acc = np.mean(val_acc)
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break
        

        # ----------------------------------
        #  Start evaluation with best model
        # ----------------------------------
        model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

        # テストデータの処理
        preds = []
        model.eval()
        for X, subject_idx in tqdm(test_loader, desc="Test"):
            X = X.to(args.device)
            subject_idx = subject_idx.to(args.device)

            with torch.no_grad():
                preds.append(model(X, subject_idx).cpu())

        preds = torch.cat(preds, dim=0).numpy()
        np.save(os.path.join(logdir, "submission"), preds)
        cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

    
if __name__ == "__main__":
    run()
