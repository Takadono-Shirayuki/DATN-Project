import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    """Simple MLP embedder for vector inputs (used with synthetic embeddings).

    Input: feature vector (dim)
    Output: normalized embedding (out_dim)
    """
    def __init__(self, in_dim=128, out_dim=128, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def embed_batch(model, X, device='cpu', batch_size=256):
    model.to(device)
    model.eval()
    xs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            b = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            e = model(b)
            xs.append(e.cpu().numpy())
    return np.vstack(xs)
