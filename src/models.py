import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        num_subjects: int,
        hid_dim: int = 128,
        p_drop: float = 0.5,
        weight_decay: float = 1e-4
    ) -> None:
        super().__init__()

        print(f"Initializing BasicConvClassifier with: num_classes={num_classes}, seq_len={seq_len}, in_channels={in_channels}, hid_dim={hid_dim}, p_drop={p_drop}, weight_decay={weight_decay}")

        # MEGデータ用のCNNブロック
        self.mega_blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim, kernel_size=3, p_drop=p_drop),
            ConvBlock(hid_dim, hid_dim, kernel_size=3, p_drop=p_drop),
        )

        self.mega_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(p_drop),
        )

        # 画像用のCNNブロック
        self.image_blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.image_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange("b c 1 1 -> b c"),
            nn.Linear(128, 64),
            nn.Dropout(p_drop),
        )

        # subject_idx用の埋め込み層と全結合層
        self.subject_embedding = nn.Embedding(num_subjects, 16)
        self.fc_subject = nn.Linear(16, 32)

        # 統合された特徴量用の全結合層
        self.final_fc_with_images = nn.Linear(hid_dim + 64 + 32, num_classes)
        self.final_fc_without_images = nn.Linear(hid_dim + 32, num_classes)

    def forward(self, X: torch.Tensor, subject_idx: torch.Tensor, images: torch.Tensor = None) -> torch.Tensor:
        # MEGデータの処理
        X = self.mega_blocks(X)
        X = self.mega_head(X)  # shape: (batch_size, hid_dim)

        # subject_idxの処理
        subject_emb = self.subject_embedding(subject_idx)
        subject_emb = F.relu(self.fc_subject(subject_emb))  # shape: (batch_size, 32)

        if images is not None:
            # 画像データの処理
            images = self.image_blocks(images)
            images = self.image_head(images)  # shape: (batch_size, 64)
            # 全ての特徴量を統合
            X = torch.cat((X, images, subject_emb), dim=1)  # shape: (batch_size, hid_dim + 64 + 32)
            X = self.final_fc_with_images(X)
        else:
            # 画像データがない場合
            X = torch.cat((X, subject_emb), dim=1)  # shape: (batch_size, hid_dim + 32)
            X = self.final_fc_without_images(X)

        return X

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p_drop):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.5,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        kernel_size = int(kernel_size)  
        padding = (kernel_size - 1) // 2  
        print(f"ConvBlock init: in_dim={in_dim}, out_dim={out_dim}, kernel_size={kernel_size}, padding={padding}, p_drop={p_drop}")

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding)
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding=padding)

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)
