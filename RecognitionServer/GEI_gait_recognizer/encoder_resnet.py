import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetEmbedder(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True, backbone='resnet18', grayscale=True):
        super().__init__()
        if backbone == 'resnet18':
            m = models.resnet18(pretrained=pretrained)
            inplanes = 64
        else:
            m = models.resnet34(pretrained=pretrained)
            inplanes = 64

        # adapt first conv to grayscale if needed
        if grayscale:
            conv1 = nn.Conv2d(1, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            # copy weights by averaging channels if pretrained
            if pretrained:
                w = m.conv1.weight.data
                w = w.mean(dim=1, keepdim=True)
                conv1.weight.data.copy_(w)
            m.conv1 = conv1

        # remove fc and avgpool
        self.backbone = nn.Sequential(*list(m.children())[:-2])
        self.avgpool = m.avgpool
        feat_dim = m.fc.in_features

        self.embedding = nn.Linear(feat_dim, embedding_dim)

    def forward(self, x):
        # x: (B,1,H,W) or (B,3,H,W)
        f = self.backbone(x)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)
        emb = self.embedding(f)
        emb = F.normalize(emb, p=2, dim=1)
        return emb
