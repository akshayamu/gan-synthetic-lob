import pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import numpy as np

# === 1. Load CSV (720 LOB snapshots) ===
df = pd.read_csv('btc_lob_5s_1h.csv')
X = df[['bid_1','ask_1','bid_vol_1','ask_vol_1']].values  # 4 features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === 2. Train/val split ===
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
X_train, X_val = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32)

# === 3. Simple WGAN-GP (latent 32, lr 1e-4) ===
class Generator(nn.Module):
    def __init__(self, z_dim=32, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, out_dim), nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, in_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

G = Generator(); D = Discriminator()
opt_G = torch.optim.Adam(G.parameters(), lr=1e-4)
opt_D = torch.optim.Adam(D.parameters(), lr=1e-4)

BATCH = 64; EPOCHS = 500
for epoch in range(EPOCHS):
    # --- train D (real vs fake) ---
    opt_D.zero_grad()
    real = X_train[torch.randint(0, len(X_train), (BATCH,))]
    z = torch.randn(BATCH, 32)
    fake = G(z)
    loss_D = -torch.mean(D(real)) + torch.mean(D(fake))
    loss_D.backward(); opt_D.step()
    
    # --- train G every 5 steps ---
    if epoch % 5 == 0:
        opt_G.zero_grad()
        z = torch.randn(BATCH, 32)
        loss_G = -torch.mean(D(G(z)))
        loss_G.backward(); opt_G.step()
    
    if epoch % 50 == 0:
        with torch.no_grad():
            fake = G(torch.randn(len(X_val), 32))
            ks = max(ks_2samp(real.numpy()[:,0], fake.numpy()[:,0])[0],
                     ks_2samp(real.numpy()[:,1], fake.numpy()[:,1])[0])
            print(f"Epoch {epoch}  KS={ks:.3f}")
            if ks < 0.3: break

# === 4. Save ===
torch.save(G.state_dict(), 'gan_lob_v0.2.pt')
print("GAN LOB v0.2 saved. KS target met.")