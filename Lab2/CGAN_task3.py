import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tqdm
import os

# ===================== HYPERPARAMETERS =====================
mb_size = 64
Z_dim = 100
h_dim = 128
X_dim = 784
num_classes = 10
lr = 1e-3
epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== DATA =====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST(root='./MNIST', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=mb_size, shuffle=True)

# ===================== HELPERS =====================
def one_hot(labels, num_classes=10):
    return F.one_hot(labels, num_classes).float()

def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

# ===================== GENERATOR =====================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(Z_dim + num_classes, h_dim)
        self.fc2 = nn.Linear(h_dim, X_dim)
        self.apply(xavier_init)

    def forward(self, z, y):
        x = torch.cat([z, y], dim=1)
        h = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h))

# ===================== DISCRIMINATOR =====================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(X_dim + num_classes, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)
        self.apply(xavier_init)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        h = F.relu(self.fc1(x))
        return self.fc2(h)  # logits

# ===================== SAVE SAMPLES =====================
def save_samples(G, epoch, digit):
    G.eval()
    z = torch.randn(16, Z_dim).to(device)
    labels = torch.full((16,), digit).to(device)
    y = one_hot(labels).to(device)

    with torch.no_grad():
        samples = G(z, y).cpu().numpy()

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(sample.reshape(28, 28), cmap='gray')

    os.makedirs("cgan_samples", exist_ok=True)
    plt.savefig(f"cgan_samples/epoch_{epoch}_digit_{digit}.png")
    plt.close()

# ===================== TRAINING =====================
def train():
    G = Generator().to(device)
    D = Discriminator().to(device)

    G_solver = optim.Adam(G.parameters(), lr=lr)
    D_solver = optim.Adam(D.parameters(), lr=lr)

    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        G.train()
        D.train()

        for X_real, labels in tqdm.tqdm(train_loader):
            X_real = X_real.to(device)
            labels = labels.to(device)

            y = one_hot(labels).to(device)

            batch_size = X_real.size(0)

            real_targets = torch.ones(batch_size, 1).to(device)
            fake_targets = torch.zeros(batch_size, 1).to(device)

            # ================= DISCRIMINATOR =================
            z = torch.randn(batch_size, Z_dim).to(device)
            G_sample = G(z, y)

            D_real = D(X_real, y)
            D_fake = D(G_sample.detach(), y)

            D_loss_real = loss_fn(D_real, real_targets)
            D_loss_fake = loss_fn(D_fake, fake_targets)
            D_loss = D_loss_real + D_loss_fake

            D_solver.zero_grad()
            D_loss.backward()
            D_solver.step()

            # ================= GENERATOR =================
            z = torch.randn(batch_size, Z_dim).to(device)
            G_sample = G(z, y)

            D_fake = D(G_sample, y)

            G_loss = loss_fn(D_fake, real_targets)

            G_solver.zero_grad()
            G_loss.backward()
            G_solver.step()

        print(f"Epoch {epoch}: D_loss={D_loss.item():.4f}, G_loss={G_loss.item():.4f}")

        # Save samples for digit 3 (example)
        save_samples(G, epoch, digit=3)

# ===================== RUN =====================
if __name__ == "__main__":
    train()