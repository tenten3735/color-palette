import torch
import torch.nn as nn

class ColorPaletteNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 15),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def summary(self, input_size=(8,)):
        from torchsummary import summary
        summary(self.model, input_size)

