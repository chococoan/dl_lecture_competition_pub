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

        
        self.subject_embedding = nn.Embedding(num_subjects, 16)
        self.fc_subject = nn.Linear(16, 32)

       
        self.final_fc = nn.Linear(hid_dim + 32, num_classes)

    def forward(self, X: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        
        X = self.mega_blocks(X)
        X = self.mega_head(X)  

        subject_emb = self.subject_embedding(subject_idx)
        subject_emb = F.relu(self.fc_subject(subject_emb))  

        X = torch.cat((X, subject_emb), dim=1)  
        X = self.final_fc(X)

        return X



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

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding='same')
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding='same')

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
