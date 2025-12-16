import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# Load data
# -----------------------
X = np.load("data/marginals_spread_depth.npy")
X = torch.tensor(X, dtype=torch.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
X = X.to(device)

# -----------------------
# Models
# -----------------------
class Generator(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


G = Generator().to(device)
D = Critic().to(device)

opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

# -----------------------
# Gradient penalty
# -----------------------
def gradient_penalty(D, real, fake):
    alpha = torch.rand(len(real), 1).to(device)
    interp = alpha * real + (1 - alpha) * fake
    interp.requires_grad_(True)

    out = D(interp)
    grad = torch.autograd.grad(
        outputs=out,
        inputs=interp,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True,
    )[0]

    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


# -----------------------
# Training
# -----------------------
EPOCHS = 3000
BATCH = 64
N_CRITIC = 5
LAMBDA_GP = 10

for epoch in range(EPOCHS):
    for _ in range(N_CRITIC):
        idx = torch.randint(0, len(X), (BATCH,))
        real = X[idx]

        z = torch.randn(BATCH, 16).to(device)
        fake = G(z).detach()

        loss_D = (
            -D(real).mean()
            + D(fake).mean()
            + LAMBDA_GP * gradient_penalty(D, real, fake)
        )

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

    z = torch.randn(BATCH, 16).to(device)
    loss_G = -D(G(z)).mean()

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")

# -----------------------
# Save samples
# -----------------------
with torch.no_grad():
    z = torch.randn(1000, 16).to(device)
    samples = G(z).cpu().numpy()

np.save("data/wgan_gp_marginals.npy", samples)
print("Saved WGAN-GP samples:", samples.shape)
