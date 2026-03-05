import torch
from open_set.encoder_resnet import ResNetEmbedder


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # default: resnet18, grayscale True
    m18 = ResNetEmbedder(embedding_dim=128, pretrained=False, backbone='resnet18', grayscale=True)
    t18, tr18 = count_params(m18)
    print('resnet18 (modified) total params:', t18)
    print('resnet18 (modified) trainable params:', tr18)

    m34 = ResNetEmbedder(embedding_dim=128, pretrained=False, backbone='resnet34', grayscale=True)
    t34, tr34 = count_params(m34)
    print('resnet34 (modified) total params:', t34)
    print('resnet34 (modified) trainable params:', tr34)
