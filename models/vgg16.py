from torch import nn
import torchvision.models as models

class VGGNet(nn.Module):
    def __init__(self, n_out, is_sigmoid):
        super(VGGNet, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(in_features = 4096, out_features = n_out)
        self.is_sigmoid = is_sigmoid

    def forward(self, x):
        x = self.model(x)
        if self.is_sigmoid:
            x = nn.Sigmoid()(x)
        return x
