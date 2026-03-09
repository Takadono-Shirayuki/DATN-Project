import os
import random
import argparse
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from open_set.encoder_resnet import ResNetEmbedder


class GEIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # expect root_dir/{label}/*.png
        self.items = []  # (path, label)
        labels = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        labels = sorted(labels)
        self.label_map = {l: i for i, l in enumerate(labels)}
        for l in labels:
            files = sorted(glob(os.path.join(root_dir, l, '*.png')))
            for f in files:
                self.items.append((f, self.label_map[l]))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, lbl = self.items[idx]
        img = Image.open(path).convert('L')
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, lbl


def make_triplets(batch_labels):
    # batch_labels: list of ints, length N
    idx_per_label = {}
    for i, l in enumerate(batch_labels):
        idx_per_label.setdefault(l, []).append(i)

    anchors, positives, negatives = [], [], []
    N = len(batch_labels)
    for i, l in enumerate(batch_labels):
        pos_choices = idx_per_label.get(l, [])
        if len(pos_choices) <= 1:
            continue
        pos = random.choice([p for p in pos_choices if p != i])
        neg_label = random.choice([ll for ll in idx_per_label.keys() if ll != l])
        neg = random.choice(idx_per_label[neg_label])
        anchors.append(i)
        positives.append(pos)
        negatives.append(neg)
    return anchors, positives, negatives


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = GEIDataset(args.gei_root, transform=transform)
    if len(dataset) == 0:
        print('No GEI images found in', args.gei_root)
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = ResNetEmbedder(embedding_dim=args.embedding_dim, pretrained=args.pretrained, grayscale=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.TripletMarginLoss(margin=args.margin)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.numpy().tolist()
            emb = model(imgs)

            a_idx, p_idx, n_idx = make_triplets(labels)
            if len(a_idx) == 0:
                continue
            anc = emb[a_idx]
            pos = emb[p_idx]
            neg = emb[n_idx]

            loss = criterion(anc, pos, neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / (len(loader) if len(loader)>0 else 1)
        print(f'Epoch {epoch+1}/{args.epochs} - loss: {avg:.4f}')

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print('Saved model to', args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gei_root', type=str, default='gait/GEI')
    parser.add_argument('--save_path', type=str, default='open_set/encoder_resnet.pth')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()
    train(args)
