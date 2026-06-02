"""
Registry DL — Script 02b: Masked Next-Cancer Predictor

Trains a masked prediction model (cancer BERT style):
  - Input:  patient's known cancer sites (one site zeroed out) + age + sex
  - Target: the masked site
  - Loss:   BCEWithLogitsLoss on the masked position only

This directly answers: "given your current cancers and demographics,
what is the probability of each additional cancer site?"

Training population: multi-cancer patients only (n≈4,052) — each patient
with k sites contributes k leave-one-out training examples.

Outputs:
  models/predictor_weights.pt
  data/predictor_pids.npy       — pid order aligned with training data
  results/02b_predictor/fig_train_loss.png
  results/02b_predictor/fig_recall_at_k.png
  results/02b_predictor/val_predictions.csv  — pid, masked_site, rank_of_true, top5
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
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print("ERROR: PyTorch not found. pip install torch")
    raise SystemExit(1)

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_DIM  = 256
DROPOUT     = 0.25
BATCH_SIZE  = 128
LR          = 5e-4
MAX_EPOCHS  = 200
PATIENCE    = 20
VAL_FRAC    = 0.20
SEED        = 42

BASE  = Path(__file__).parent.parent
DOUT  = BASE / "data"
MOUT  = BASE / "models"
OUT   = BASE / "results/02b_predictor"
OUT.mkdir(parents=True, exist_ok=True)
MOUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────

class MaskedCancerDataset(Dataset):
    """
    For each multi-cancer patient with k sites, generates k examples:
      x = sites with site_i zeroed out + [age_norm, sex_binary]
      y_mask = one-hot of site_i (the masked site)
      weight = 1.0 (all positions equal)
    """
    def __init__(self, X, meta, sites, pids):
        self.examples = []
        age_mean = meta["age_first"].mean()
        age_std  = meta["age_first"].std() + 1e-6

        for pid in pids:
            row   = X.loc[pid]
            age_n = (meta.loc[pid, "age_first"] - age_mean) / age_std
            sex_b = 1.0 if meta.loc[pid, "sex"] == "M" else 0.0
            active = [i for i, s in enumerate(sites) if row[s] == 1]
            if len(active) < 2:
                continue
            for mask_idx in active:
                x = row.values.astype(np.float32).copy()
                x[mask_idx] = 0.0          # zero out the masked site
                x = np.append(x, [age_n, sex_b]).astype(np.float32)
                y = np.zeros(len(sites), dtype=np.float32)
                y[mask_idx] = 1.0
                self.examples.append((x, y, mask_idx))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y, mask_idx = self.examples[idx]
        return torch.tensor(x), torch.tensor(y), mask_idx


# ── Model ─────────────────────────────────────────────────────────────────────

class CancerMaskedPredictor(nn.Module):
    def __init__(self, n_sites, hidden_dim, dropout):
        super().__init__()
        in_dim = n_sites + 2   # sites + age_norm + sex
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_sites),   # logits over all sites
        )

    def forward(self, x):
        return self.net(x)   # shape: (batch, n_sites)


def masked_bce_loss(logits, y_onehot, mask_indices):
    """Compute BCE only at the masked positions."""
    batch_idx = torch.arange(logits.size(0))
    logits_at_mask = logits[batch_idx, mask_indices]
    target_at_mask = y_onehot[batch_idx, mask_indices]
    return nn.functional.binary_cross_entropy_with_logits(
        logits_at_mask, target_at_mask, reduction="mean")


# ── Train / Eval ──────────────────────────────────────────────────────────────

def recall_at_k(logits_np, true_mask_indices, k_list=(1, 3, 5)):
    """Fraction of examples where true site is in top-k predicted sites."""
    probs  = 1 / (1 + np.exp(-logits_np))   # sigmoid
    ranks  = np.argsort(-probs, axis=1)      # descending
    results = {}
    for k in k_list:
        top_k = ranks[:, :k]
        hits  = [true_mask_indices[i] in top_k[i] for i in range(len(true_mask_indices))]
        results[f"R@{k}"] = np.mean(hits)
    return results


def main():
    print("=== Registry DL — 02b: Masked Next-Cancer Predictor ===")

    X_df  = pd.read_csv(DOUT / "cancer_matrix.csv", index_col="pid")
    meta  = pd.read_csv(DOUT / "patient_meta.csv",  index_col="pid")
    sites = X_df.columns.tolist()
    N_S   = len(sites)

    # Multi-cancer patients only
    multi_pids = meta[meta["n_sites"] >= 2].index.tolist()
    print(f"  Multi-cancer patients: {len(multi_pids):,} / {len(X_df):,}")

    # Train/val split at patient level (not example level — prevent leakage)
    rng      = np.random.default_rng(SEED)
    shuffled = rng.permutation(multi_pids)
    n_val    = int(len(shuffled) * VAL_FRAC)
    val_pids   = list(shuffled[:n_val])
    train_pids = list(shuffled[n_val:])

    ds_train = MaskedCancerDataset(X_df, meta, sites, train_pids)
    ds_val   = MaskedCancerDataset(X_df, meta, sites, val_pids)
    print(f"  Train examples: {len(ds_train):,}  Val examples: {len(ds_val):,}")

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False)

    model = CancerMaskedPredictor(N_S, HIDDEN_DIM, DROPOUT)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    history = []
    best_val, patience_ctr = float("inf"), 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb, mask_idx in dl_train:
            opt.zero_grad()
            logits = model(xb)
            loss   = masked_bce_loss(logits, yb, mask_idx)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, mask_idx in dl_val:
                logits   = model(xb)
                val_loss += masked_bce_loss(logits, yb, mask_idx).item()

        tr_loss  /= len(dl_train)
        val_loss /= len(dl_val)
        history.append({"epoch": epoch, "train": tr_loss, "val": val_loss})

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={tr_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val - 1e-5:
            best_val, patience_ctr = val_loss, 0
            torch.save(model.state_dict(), MOUT / "predictor_weights.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop epoch {epoch}")
                break

    # ── Evaluation on val set ─────────────────────────────────────────────────
    model.load_state_dict(torch.load(MOUT / "predictor_weights.pt"))
    model.eval()

    all_logits, all_mask_idx, all_true_sites = [], [], []
    with torch.no_grad():
        for xb, yb, mask_idx in dl_val:
            logits = model(xb)
            all_logits.append(logits.numpy())
            all_mask_idx.extend(mask_idx.tolist())

    all_logits_np = np.concatenate(all_logits, axis=0)
    rec = recall_at_k(all_logits_np, all_mask_idx)
    print(f"\n  Validation Recall@k:")
    for k, v in rec.items():
        print(f"    {k}: {v:.3f}")

    # Save val predictions
    probs  = 1 / (1 + np.exp(-all_logits_np))
    ranks  = np.argsort(-probs, axis=1)
    rows = []
    for i, (mi, logit_row, rank_row) in enumerate(zip(all_mask_idx, all_logits_np, ranks)):
        true_site = sites[mi]
        rank_of_true = int(np.where(rank_row == mi)[0][0]) + 1
        top5 = [sites[j] for j in rank_row[:5]]
        rows.append({"masked_site": true_site, "rank_of_true": rank_of_true,
                     "top5_predicted": ", ".join(top5)})
    pd.DataFrame(rows).to_csv(OUT / "val_predictions.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    hist_df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist_df["epoch"], hist_df["train"], label="train")
    ax.plot(hist_df["epoch"], hist_df["val"],   label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Masked BCE loss")
    ax.set_title("Masked next-cancer predictor — training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_train_loss.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    ks = list(rec.keys())
    vs = [rec[k] for k in ks]
    bars = ax.bar(ks, vs, color=["#14304a","#2e7fbf","#7eb8df"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title(f"Next-cancer prediction accuracy\n(val n={len(all_mask_idx):,} examples)")
    for b, v in zip(bars, vs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_recall_at_k.png", dpi=150)
    plt.close()

    print(f"\n  Best val loss: {best_val:.5f}")
    print(f"  Saved → models/predictor_weights.pt")
    print(f"  Figures → results/02b_predictor/")
    return rec


if __name__ == "__main__":
    main()
