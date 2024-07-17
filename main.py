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
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from torchvision import transforms




def preprocess_data_batch(data_batch, original_fs=200, target_fs=100, seed=1234):
    set_seed(seed)  # シードを設定

    # データ型の変換
    data_batch = data_batch.astype(np.float32)
    
    with ThreadPoolExecutor() as executor:
        # 並列化して前処理を行う
        futures = [executor.submit(preprocess_single_sample, data, original_fs, target_fs) for data in data_batch]
        results = [future.result() for future in futures]

    return np.array(results)

def preprocess_single_sample(data, original_fs, target_fs, 
                             resample=True, filter=True, scale=True, baseline=True, window_size=10):
    if resample:
        data = resample_data(data, original_fs, target_fs)
    if filter:
        data = bandpass_filter(data, lowcut=1, highcut=40, fs=target_fs)
    if scale:
        data = scale_data(data)
    if baseline:
        data = baseline_correction(data, window_size=10)
    return data

def resample_data(data, original_fs, target_fs):
    num_samples = int(data.shape[1] * target_fs / original_fs)  # サンプル数を整数に変換
    data_resampled = signal.resample(data, num_samples, axis=1)
    return data_resampled

def bandpass_filter(data, lowcut, highcut, fs):
    order = 4
    b, a = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    if any(s < 0 for s in filtered_data.strides):
        filtered_data = filtered_data.copy()
    return filtered_data


def scale_data(data):
    scaler = StandardScaler()
    # スケーリング前のデータの形状を維持
    data_reshaped = data.reshape(-1, data.shape[-1])
    data_scaled = scaler.fit_transform(data_reshaped).reshape(data.shape)
    return data_scaled

def baseline_correction(data, window_size):
    window_size = int(window_size)  
    corrected_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        baseline = np.convolve(data[i], np.ones(window_size)/window_size, mode='same')
        corrected_data[i] = data[i] - baseline
    return corrected_data

class ThingsMEGDatasetOnTheFly(ThingsMEGDataset):
    def __init__(self, split: str, data_dir: str = "data", original_fs=200, target_fs=100, 
                 resample=True, filter=True, scale=True, baseline=True) -> None:
        super().__init__(split, data_dir)
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.split = split
        self.resample = resample
        self.filter = filter
        self.scale = scale
        self.baseline = baseline

    def __getitem__(self, i):
        data = self.X[i]
        data = preprocess_single_sample(data.numpy(), self.original_fs, self.target_fs, 
                                        self.resample, self.filter, self.scale, self.baseline)
        data = torch.tensor(data, dtype=torch.float32)
        subject_idx = self.subject_idxs[i]  # subject_idxを取得
        if self.split == 'test':
            return data, subject_idx
        else:
            label = self.y[i]
            return data, label, subject_idx

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths_file, base_dir, transform=None):
        with open(image_paths_file, 'r') as f:
            self.image_paths = [os.path.join(base_dir, line.strip()) for line in f]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 画像を224x224ピクセルにリサイズ
    transforms.ToTensor(),  # 画像をPyTorchテンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNetの平均と標準偏差で正規化
])

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    try:
        print("Setting seed...")
        set_seed(1234)
        logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        print(f"Log directory: {logdir}")
        if args.use_wandb:
            wandb.init(mode="online", dir=logdir, project="MEG-classification")

        # ベースディレクトリ（画像が保存されているディレクトリ）
        image_base_dir = '/content/drive/MyDrive/ColabNotebooks/DLBasics2024_colab/lastsubmission/dl_lecture_competition_pub/data/Images'
        
        # パスファイルのディレクトリ
        data_base_dir = '/content/drive/MyDrive/ColabNotebooks/DLBasics2024_colab/lastsubmission/dl_lecture_competition_pub/data'
        train_image_paths_file = os.path.join(data_base_dir, 'train_image_paths.txt')
        val_image_paths_file = os.path.join(data_base_dir, 'val_image_paths.txt')

        # 画像データセットとデータローダーの作成
        train_image_dataset = ImageDataset(train_image_paths_file, image_base_dir, transform=image_transforms)
        val_image_dataset = ImageDataset(val_image_paths_file, image_base_dir, transform=image_transforms)
        
        image_loader_args = {"batch_size": 64, "num_workers": args.num_workers}

        train_image_loader = torch.utils.data.DataLoader(train_image_dataset, shuffle=True, **image_loader_args)
        val_image_loader = torch.utils.data.DataLoader(val_image_dataset, shuffle=False, **image_loader_args)

        print("Loading MEG datasets...")
        train_set = ThingsMEGDatasetOnTheFly("train", args.data_dir, resample=True, filter=False, scale=False, baseline=False)
        val_set = ThingsMEGDatasetOnTheFly("val", args.data_dir, resample=True, filter=False, scale=False, baseline=False)
        test_set = ThingsMEGDatasetOnTheFly("test", args.data_dir, resample=True, filter=False, scale=False, baseline=False)

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

        # データの形状を確認
        for i, ((X, y, subject_idx), images) in enumerate(zip(train_loader, train_image_loader)):
            print(f"MEGデータの形状: {X.shape}")
            print(f"ラベルの形状: {y.shape}")
            print(f"subject_idxの形状: {subject_idx.shape}")
            print(f"画像データの形状: {images.shape}")
            if i == 0:
                break  # 一度だけ形状を確認するためにループを抜ける

        for epoch in range(args.epochs):
            train_loss, train_acc, val_loss, val_acc = [], [], [], []

            model.train()
            for (X, y, subject_idx), images in tqdm(zip(train_loader, train_image_loader), desc="Train"):
                X = X.float().to(args.device)
                y = y.to(args.device)
                subject_idx = subject_idx.to(args.device)
                images = images.to(args.device)

                y_pred = model(X, subject_idx, images)  # モデルが画像を受け取るように調整

                loss = F.cross_entropy(y_pred, y)
                train_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = accuracy(y_pred, y)
                train_acc.append(acc.item())

            model.eval()
            for (X, y, subject_idx), images in tqdm(zip(val_loader, val_image_loader), desc="Validation"):
                X = X.float().to(args.device)
                y = y.to(args.device)
                subject_idx = subject_idx.to(args.device)
                images = images.to(args.device)

                with torch.no_grad():
                    y_pred = model(X, subject_idx, images)  # モデルが画像を受け取るように調整

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

        model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

        # テストデータの処理
        preds = []
        model.eval()
        for X, subject_idx in tqdm(test_loader, desc="Test"):
            X = X.float().to(args.device)
            subject_idx = subject_idx.to(args.device)

            with torch.no_grad():
                output = model(X, subject_idx).cpu()
                preds.append(output)

        # 予測ラベルの整形と保存
        preds = torch.cat(preds, dim=0).numpy()
        pred_labels = np.argmax(preds, axis=1)  # 予測ラベルを取得

        submission_file_path = os.path.join(logdir, "submission.npy")
        np.save(submission_file_path, pred_labels)
        cprint(f"Submission {pred_labels.shape} saved at {submission_file_path}", "cyan")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run()
