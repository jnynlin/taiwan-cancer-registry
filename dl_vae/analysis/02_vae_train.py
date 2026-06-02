"""
Registry DL — Script 02: VAE Training

Trains a Variational Autoencoder on the binary cancer co-occurrence matrix.
Architecture: binary input → encoder → (μ, logvar) → reparameterise → decoder → sigmoid

Key design choices:
  - BCEWithLogitsLoss for reconstruction (binary data)
  - β-VAE with β=1 (standard ELBO); set β>1 for more disentangled axes
  - Latent dim=12: enough to capture 3–5 real axes, rest collapse to prior (KL≈0)
  - Training stops on validation loss plateau (patience=20)

Outputs:
  models/vae_weights.pt        — trained model weights
  data/latent_mu.npy           — (N, LATENT_DIM) mean vectors (use for downstream)
  data/latent_z.npy            — (N, LATENT_DIM) sampled z vectors
  data/train_history.csv       — epoch-level loss log
  results/02_vae/fig_loss.png
  results/02_vae/fig_kl_per_dim.png
  results/02_vae/fig_recon_per_site.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("ERROR: PyTorch not found. Install with: pip install torch")
    raise SystemExit(1)

# ── Hyperparameters ────────────────────────────────────────────────────────────
LATENT_DIM  = 12
HIDDEN_DIM  = 128
BETA        = 1.0       # KL weight; β>1 → more disentangled axes
BATCH_SIZE  = 256
LR          = 1e-3
MAX_EPOCHS  = 300
PATIENCE    = 25        # early stopping
VAL_FRAC    = 0.15
SEED        = 42

BASE   = Path(__file__).parent.parent
DOUT   = BASE / "data"
MOUT   = BASE / "models"
OUT    = BASE / "results/02_vae"
OUT.mkdir(parents=True, exist_ok=True)
MOUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Model ─────────────────────────────────────────────────────────────────────

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),   # logits — no sigmoid here
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at eval

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


def elbo_loss(recon_logits, x, mu, logvar, beta):
    recon = nn.functional.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum")
    kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon + beta * kl) / x.size(0), recon / x.size(0), kl / x.size(0)


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    print("=== Registry DL — 02: VAE Training ===")

    X_df = pd.read_csv(DOUT / "cancer_matrix.csv", index_col="pid")
    X    = torch.tensor(X_df.values, dtype=torch.float32)
    N, D = X.shape
    print(f"  Matrix: {N:,} patients × {D} sites")

    # Train / val split
    n_val   = int(N * VAL_FRAC)
    n_train = N - n_val
    ds_full = TensorDataset(X)
    ds_train, ds_val = random_split(ds_full, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False)

    model  = VAE(D, HIDDEN_DIM, LATENT_DIM)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    history = []
    best_val, patience_ctr = float("inf"), 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_loss = tr_recon = tr_kl = 0.0
        for (xb,) in dl_train:
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, recon_l, kl_l = elbo_loss(recon, xb, mu, logvar, BETA)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss  += loss.item()
            tr_recon += recon_l.item()
            tr_kl    += kl_l.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (xb,) in dl_val:
                recon, mu, logvar = model(xb)
                loss, _, _ = elbo_loss(recon, xb, mu, logvar, BETA)
                val_loss += loss.item()

        tr_loss  /= len(dl_train)
        val_loss /= len(dl_val)
        sched.step(val_loss)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
                         "recon": tr_recon / len(dl_train), "kl": tr_kl / len(dl_train)})

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={tr_loss:.3f} val={val_loss:.3f}")

        # Early stopping
        if val_loss < best_val - 1e-4:
            best_val, patience_ctr = val_loss, 0
            torch.save(model.state_dict(), MOUT / "vae_weights.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
                break

    # Load best weights and extract latent vectors
    model.load_state_dict(torch.load(MOUT / "vae_weights.pt"))
    model.eval()
    with torch.no_grad():
        recon_all, mu_all, logvar_all = model(X)
        z_all = model.reparameterise(mu_all, logvar_all)

    mu_np = mu_all.numpy()
    z_np  = z_all.numpy()
    np.save(DOUT / "latent_mu.npy",  mu_np)
    np.save(DOUT / "latent_z.npy",   z_np)
    print(f"  Saved latent μ: {mu_np.shape}")

    # Reconstruction accuracy per site
    probs = torch.sigmoid(recon_all).numpy()
    pred  = (probs > 0.5).astype(int)
    X_np  = X.numpy()
    site_acc = (pred == X_np).mean(axis=0)
    site_df  = pd.DataFrame({"site": X_df.columns, "recon_acc": site_acc,
                              "prevalence": X_np.mean(axis=0)})
    site_df.to_csv(OUT / "site_recon_acc.csv", index=False)

    # Per-dimension KL (active dimensions have KL >> 0)
    kl_per_dim = (-0.5 * (1 + logvar_all - mu_all.pow(2) - logvar_all.exp())).mean(0).detach().numpy()

    # History
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUT / "train_history.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="train")
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"],   label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("ELBO loss")
    axes[0].set_title("Training curve"); axes[0].legend()
    axes[1].plot(hist_df["epoch"], hist_df["recon"], label="recon BCE", color="#2e7fbf")
    axes[1].plot(hist_df["epoch"], hist_df["kl"],    label="KL",        color="#e05c2e")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss component")
    axes[1].set_title("Reconstruction vs KL"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_loss.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 3))
    dims = np.arange(LATENT_DIM)
    colors = ["#2e7fbf" if k > 0.1 else "#aaaaaa" for k in kl_per_dim]
    ax.bar(dims, kl_per_dim, color=colors)
    ax.axhline(0.1, color="red", linestyle="--", linewidth=0.8, label="active threshold")
    ax.set_xlabel("Latent dimension"); ax.set_ylabel("Mean KL")
    ax.set_title(f"Per-dimension KL (β={BETA}) — blue = active axis")
    ax.set_xticks(dims); ax.set_xticklabels([f"z{i}" for i in dims])
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_kl_per_dim.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 4))
    order = site_df.sort_values("prevalence", ascending=False)
    ax.scatter(range(len(order)), order["recon_acc"], c=order["prevalence"],
               cmap="Blues", s=40, zorder=3)
    ax.axhline(0.95, color="green", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order["site"], rotation=90, fontsize=8)
    ax.set_ylabel("Reconstruction accuracy")
    ax.set_title("Per-site reconstruction accuracy (color = prevalence)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_recon_per_site.png", dpi=150)
    plt.close()

    n_active = (kl_per_dim > 0.1).sum()
    print(f"  Active latent dimensions (KL > 0.1): {n_active} / {LATENT_DIM}")
    print(f"  Best val ELBO: {best_val:.4f}")
    print(f"  Figures saved → results/02_vae/")


if __name__ == "__main__":
    train()
