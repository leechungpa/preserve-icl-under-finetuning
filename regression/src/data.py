import torch
from torch import nn


def get_promt(x, y):
    device = x.device
    b, n, d =  x.shape

    z = torch.cat([
            torch.ones(b, n, 1, device=device),
            x,
            y.unsqueeze(2)
        ], dim=2)

    y_test = z[:,-1,-1].clone()
    z[:,-1,-1] = torch.zeros(b, device=device)

    return z, y_test


def generate_data(b=1000, n=20, d=5, theta=None, x_std=1, noise_std=0, use_prompt=False, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if theta is None:
        theta = torch.randn(b, d, device=device)
        theta = theta / theta.norm(dim=1, keepdim=True).clamp_min(1e-12)

    x = torch.FloatTensor(b, n, d).normal_(0, x_std).to(device)
    if theta.dim() == 1:
        y = torch.einsum('d,bnd->bn', (theta, x))
    elif theta.dim() == 2:
        y = torch.einsum('bd,bnd->bn', (theta, x))

    if noise_std > 0:
        y = y + torch.randn_like(y)*noise_std

    if use_prompt:
        return get_promt(x, y)
    else:
        return x, y


def random_sample_data(x_dataset, y_dataset, b=1000, n=20, use_prompt=False, device=None):
    "x_dataset: (dataset_size, d), y_dataset: (dataset_size)"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset_size, d, = x_dataset.shape
    dataset_size, = y_dataset.shape

    idx = torch.randint(0, dataset_size, (b, n))

    x = x_dataset[idx]
    y = y_dataset[idx]

    if use_prompt:
        return get_promt(x, y)
    else:
        return x, y