"""
Registry DL — Script 05: Rare Syndrome Autoencoder

Trains a bottleneck autoencoder on the full patient matrix (all 78k patients,
including single-cancer). Patients whose cancer combination is poorly
reconstructed have anomalous multi-site patterns inconsistent with the
common co-occurrence structure — candidates for undiagnosed hereditary
cancer syndromes (Lynch, BRCA2, MEN1, Li-Fraumeni, VHL, FAP).

Key design choices:
  - All patients in training (including single-cancer) so the model learns
    the NORMAL distribution; deviation from normal = anomaly
  - Bottleneck dim=16: tight enough to force compression, generous enough
    not to lose rare but real patterns
  - Anomaly score = per-patient sum of BCE (not mean) — favours patients
    with multiple surprising sites over patients with one slightly wrong site
  - Top 1% threshold (~784 patients) flagged for syndrome screening

Outputs:
  models/autoencoder_weights.pt
  results/05_anomaly/anomaly_scores.csv   — pid, score, rank, n_sites, sites
  results/05_anomaly/fig_score_dist.png
  results/05_anomaly/fig_top_candidates.png
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
except ImportError:
    print("ERROR: pip install torch"); raise SystemExit(1)

BOTTLENECK  = 16
HIDDEN_DIM  = 64
BATCH_SIZE  = 512
LR          = 1e-3
MAX_EPOCHS  = 300
PATIENCE    = 25
VAL_FRAC    = 0.10
TOP_FRAC    = 0.01    # top 1% flagged as anomalous
SEED        = 42

BASE  = Path(__file__).parent.parent
DOUT  = BASE / "data"
MOUT  = BASE / "models"
OUT   = BASE / "results/05_anomaly"
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED)


class CancerAutoencoder(nn.Module):
    def __init__(self, n_sites, hidden, bottleneck):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_sites, hidden),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_sites),   # logits
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def main():
    print("=== Registry DL — 05: Rare Syndrome Autoencoder ===")

    X_df  = pd.read_csv(DOUT / "cancer_matrix.csv", index_col="pid")
    meta  = pd.read_csv(DOUT / "patient_meta.csv",  index_col="pid")
    sites = X_df.columns.tolist()
    X     = torch.tensor(X_df.values, dtype=torch.float32)
    N, D  = X.shape
    print(f"  Matrix: {N:,} patients × {D} sites  (all patients, incl. single-cancer)")

    n_val   = int(N * VAL_FRAC)
    ds      = TensorDataset(X)
    ds_tr, ds_val = random_split(ds, [N - n_val, n_val],
                                  generator=torch.Generator().manual_seed(SEED))
    dl_tr  = DataLoader(ds_tr,  batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    model = CancerAutoencoder(D, HIDDEN_DIM, BOTTLENECK)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    best_val, patience_ctr = float("inf"), 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr = 0.0
        for (xb,) in dl_tr:
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb), xb)
            loss.backward(); opt.step()
            tr += loss.item()

        model.eval(); vl = 0.0
        with torch.no_grad():
            for (xb,) in dl_val:
                vl += nn.functional.binary_cross_entropy_with_logits(model(xb), xb).item()
        tr /= len(dl_tr); vl /= len(dl_val)
        sched.step(vl)
        history.append({"epoch": epoch, "train": tr, "val": vl})

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={tr:.4f} val={vl:.4f}")

        if vl < best_val - 1e-5:
            best_val, patience_ctr = vl, 0
            torch.save(model.state_dict(), MOUT / "autoencoder_weights.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop epoch {epoch}"); break

    # ── Per-patient anomaly scores ────────────────────────────────────────────
    model.load_state_dict(torch.load(MOUT / "autoencoder_weights.pt"))
    model.eval()
    with torch.no_grad():
        logits  = model(X)
        # Per-patient sum-BCE (not mean) — surprises accumulate across sites
        per_site_bce = nn.functional.binary_cross_entropy_with_logits(
            logits, X, reduction="none")          # (N, D)
        scores = per_site_bce.sum(dim=1).numpy()  # (N,)

    # Identify which sites contribute most to each patient's anomaly score
    per_site_np = per_site_bce.numpy()

    rows = []
    for i, pid in enumerate(X_df.index):
        active = [sites[j] for j in range(D) if X_df.iloc[i, j] == 1]
        top_surprise = sites[int(per_site_np[i].argmax())]
        rows.append({"pid": pid, "anomaly_score": scores[i],
                     "n_sites": len(active),
                     "sites": "+".join(active) if active else "",
                     "top_surprise_site": top_surprise})

    score_df = pd.DataFrame(rows).sort_values("anomaly_score", ascending=False)
    score_df["rank"] = range(1, len(score_df) + 1)
    score_df.to_csv(OUT / "anomaly_scores.csv", index=False)

    n_flag = int(N * TOP_FRAC)
    flagged = score_df.head(n_flag)
    print(f"  Top {TOP_FRAC*100:.0f}% flagged: {n_flag:,} patients")
    print(f"  Score range (flagged): {flagged['anomaly_score'].min():.3f} – "
          f"{flagged['anomaly_score'].max():.3f}")
    print(f"  Multi-cancer among flagged: "
          f"{(flagged['n_sites'] >= 2).sum()} / {n_flag} "
          f"({(flagged['n_sites'] >= 2).mean()*100:.1f}%)")

    # ── Figures ───────────────────────────────────────────────────────────────
    hist_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist_df["epoch"], hist_df["train"], label="train")
    axes[0].plot(hist_df["epoch"], hist_df["val"],   label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Autoencoder training curve"); axes[0].legend()

    axes[1].hist(scores, bins=80, color="#2e7fbf", alpha=0.7, edgecolor="none")
    thresh = score_df.iloc[n_flag - 1]["anomaly_score"]
    axes[1].axvline(thresh, color="#e05c2e", linestyle="--",
                    label=f"Top {TOP_FRAC*100:.0f}% threshold ({thresh:.2f})")
    axes[1].set_xlabel("Anomaly score (sum-BCE per patient)")
    axes[1].set_ylabel("Patients"); axes[1].set_title("Anomaly score distribution")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_score_dist.png", dpi=150); plt.close()

    # Top candidates: n_sites distribution + top surprise sites
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    vc = flagged["n_sites"].value_counts().sort_index()
    axes[0].bar(vc.index, vc.values, color="#14304a")
    axes[0].set_xlabel("Number of cancer sites"); axes[0].set_ylabel("Flagged patients")
    axes[0].set_title(f"Site count in top {TOP_FRAC*100:.0f}% anomalous patients")

    surprise_vc = flagged["top_surprise_site"].value_counts().head(15)
    axes[1].barh(range(len(surprise_vc)), surprise_vc.values[::-1], color="#2e7fbf")
    axes[1].set_yticks(range(len(surprise_vc)))
    axes[1].set_yticklabels(surprise_vc.index[::-1], fontsize=9)
    axes[1].set_xlabel("Count"); axes[1].set_title("Most surprising site per flagged patient")
    fig.tight_layout()
    fig.savefig(OUT / "fig_top_candidates.png", dpi=150); plt.close()

    print(f"  Figures → results/05_anomaly/")
    return score_df, flagged


if __name__ == "__main__":
    main()
