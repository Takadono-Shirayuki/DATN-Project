import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure local package modules are importable when running this script directly
sys.path.insert(0, os.path.dirname(__file__))
from data_gen import generate_gait_data
from encoder import Embedder
from open_set_matcher import OpenSetGaitMatcher


class VecDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_batch(batch):
    X = np.stack([b[0] for b in batch])
    y = np.array([b[1] for b in batch])
    return torch.from_numpy(X).float(), torch.from_numpy(y).long()


def train(seed=42, n_classes=10, dim=128, n_samples_per_class=40, epochs=5, lr=1e-3, device='cpu'):
    np.random.seed(seed)
    random.seed(seed)

    g_X, g_y, q_k_X, q_k_y, q_u_X = generate_gait_data(n_classes=n_classes, dim=dim,
                                                       n_samples_per_class=n_samples_per_class, seed=seed)

    ds = VecDataset(g_X, g_y)
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate_batch)

    model = Embedder(in_dim=dim, out_dim=128, hidden=256)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=0.3)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            # form triplets within batch: for each anchor pick positive and negative randomly
            emb = model(Xb)
            batch_size = emb.size(0)
            anchors = []
            positives = []
            negatives = []
            for i in range(batch_size):
                ai = i
                label = yb[i].item()
                pos_idx = i
                # find a different positive in batch if possible
                pos_candidates = (yb == label).nonzero(as_tuple=False).view(-1).cpu().numpy().tolist()
                if len(pos_candidates) > 1:
                    pos_idx = random.choice([p for p in pos_candidates if p != i])
                else:
                    pos_idx = i

                neg_candidates = (yb != label).nonzero(as_tuple=False).view(-1).cpu().numpy().tolist()
                if len(neg_candidates) == 0:
                    neg_idx = i
                else:
                    neg_idx = random.choice(neg_candidates)

                anchors.append(emb[ai].unsqueeze(0))
                positives.append(emb[pos_idx].unsqueeze(0))
                negatives.append(emb[neg_idx].unsqueeze(0))

            if len(anchors) == 0:
                continue

            A = torch.cat(anchors, dim=0)
            P = torch.cat(positives, dim=0)
            N = torch.cat(negatives, dim=0)

            loss = criterion(A, P, N)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {ep+1}/{epochs} loss: {total_loss/len(loader):.4f}')

    # Save model
    os.makedirs('open_set/models', exist_ok=True)
    torch.save(model.state_dict(), 'open_set/models/encoder.pth')

    # Evaluate: embed gallery and queries, fit matcher and compute accuracy
    model.eval()
    with torch.no_grad():
        Xg = torch.from_numpy(g_X).float().to(device)
        Eg = model(Xg).cpu().numpy()
        Xqk = torch.from_numpy(q_k_X).float().to(device)
        Eqk = model(Xqk).cpu().numpy()
        Xqu = torch.from_numpy(q_u_X).float().to(device)
        Equ = model(Xqu).cpu().numpy()

    matcher = OpenSetGaitMatcher(metric='cosine', alpha=3.0, filename='encoder_db.json')
    matcher.fit(Eg, g_y)

    preds_k, dists_k = matcher.predict(Eqk)
    known_acc = np.mean(preds_k == q_k_y)
    preds_u, dists_u = matcher.predict(Equ)
    unknown_acc = np.mean(preds_u == -1)

    print('--- Encoder evaluation ---')
    print(f'Known acc: {known_acc*100:.2f}%, Unknown acc: {unknown_acc*100:.2f}%')


if __name__ == '__main__':
    train(epochs=5, n_samples_per_class=50)
