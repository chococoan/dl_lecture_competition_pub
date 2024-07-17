import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int, in_channels: int) -> None:
        super(EfficientNetClassifier, self).__init__()
        self.efficient_net = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # カスタマイズされた最初の畳み込み層
        self.efficient_net.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False
        )
        
        self.efficient_net.classifier[1] = nn.Linear(self.efficient_net.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficient_net(x)