import os
import torch
import torchaudio
from typing import Tuple
from termcolor import cprint

import os
import torch
import torchaudio
from typing import Tuple
from termcolor import cprint

class ThingsMEGDatasetSpectrogram(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        spectrogram = self._convert_to_spectrogram(X)
        if hasattr(self, "y"):
            return spectrogram, self.y[i], self.subject_idxs[i]
        else:
            return spectrogram, self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

    def _convert_to_spectrogram(self, X):
        spectrograms = []
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(n_fft=512, n_mels=64)
        for channel_data in X:
            spectrogram = mel_spectrogram(channel_data.unsqueeze(0))
            spectrograms.append(spectrogram)
        spectrograms = torch.stack(spectrograms).squeeze(1)
        
        # データの正規化を追加
        spectrograms = (spectrograms - spectrograms.mean()) / spectrograms.std()
        
        return spectrograms
    

