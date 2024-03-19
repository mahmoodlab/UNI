import torch
import torch.nn as nn

class TimmAvgPool(nn.Module):
    def __init__(self, encoder):
        super(TimmAvgPool, self).__init__()
        self.encoder = encoder
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x