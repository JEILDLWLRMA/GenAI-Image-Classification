import torch
import torch.nn as nn
import timm

class SwinNet(nn.Module):
    def __init__(self, img_size = 224, pretrained=True):
        super().__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=pretrained)
        num_features = self.model.head.in_features
        self.model.head = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x