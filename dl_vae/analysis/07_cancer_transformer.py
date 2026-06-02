"""
Registry DL — Script 07: Cancer Sequence Transformer (P1)

Treats each patient's cancer history as a sentence:
  token    = cancer site code (C02, C15, ...)
  position = time since first diagnosis (sinusoidal, log-day scale)
  context  = age at each diagnosis + sex (per-token)

Architecture: small BERT (2 layers, 4 heads, d=64)
Training: masked site prediction — randomly mask 15% of sites in
          multi-cancer sequences, predict the masked site.
          Single-cancer patients contribute to embedding learning
          (forward pass) but are not masked — no loss computed.

Key difference over 02b masked MLP:
  - TIME-AWARE: time gap between cancers is encoded; C12→C15 in 6 months
    vs C12→C15 in 3 years produces different context representations
  - ORDER-AWARE: C12 first then C15 ≠ C15 first then C12
  - TRIGRAMS: full sequence context, not just single-hop pairs
  - SHARED SITE EMBEDDINGS: all 84k patients shape what each site "means"

Vocab:
  0 = [PAD], 1 = [MASK], 2 = [CLS], 3..N = site codes (alphabetical)

Outputs:
  models/transformer_weights.pt
  data/patient_cls_embeddings.npy  — (N_patients, d_model) CLS vectors
  data/transformer_site_vocab.csv  — site → token index mapping
  results/07_transformer/fig_train_loss.png
  results/07_transformer/fig_recall_at_k.png
  results/07_transformer/val_predictions.csv
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
    print("ERROR: pip install torch"); raise SystemExit(1)

# ── Hyperparameters ────────────────────────────────────────────────────────────
D_MODEL     = 64
N_HEAD      = 4
N_LAYERS    = 2
FF_DIM      = 128
MAX_SEQ     = 8        # [CLS] + up to 6 cancers + 1 spare
MASK_PROB   = 0.15
DROPOUT     = 0.1
BATCH_SIZE  = 256
LR          = 1e-3
MAX_EPOCHS  = 200
PATIENCE    = 20
VAL_FRAC    = 0.20
SEED        = 42

PAD_IDX  = 0
MASK_IDX = 1
CLS_IDX  = 2
SITE_OFFSET = 3   # site tokens start at index 3

BASE  = Path(__file__).parent.parent
RAW   = BASE.parent / "data/processed/all_cancers.csv"
DOUT  = BASE / "data"
MOUT  = BASE / "models"
OUT   = BASE / "results/07_transformer"
OUT.mkdir(parents=True, exist_ok=True)
MOUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED)


# ── Data loading ──────────────────────────────────────────────────────────────

def roc_to_ts(x):
    s = str(x).split(".")[0].zfill(7)
    if not s.isdigit() or len(s) != 7: return pd.NaT
    y = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00","99") else "01"
    dd = s[5:7] if s[5:7] not in ("00","99") else "01"
    try: return pd.Timestamp(f"{y}-{mm}-{dd}")
    except: return pd.NaT


def build_sequences(min_site_n=30):
    """
    Returns:
      seqs: list of per-patient dicts with keys:
        pid, tokens (site indices), days (since first dx),
        ages, sex_bin, n_cancers
      vocab: dict site_code → token_idx (starting at SITE_OFFSET)
    """
    print("  Loading registry…")
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = df.columns.str.strip().str.lstrip('﻿')
    df["pid"]  = df["病歷號(2)"].astype(str).str.strip()
    df["site"] = df["腫瘤部位(47)"].astype(str).str[:3].str.upper()
    df["dx"]   = df["最初診斷日(45)"].apply(roc_to_ts)
    df["age"]  = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["sex"]  = df["性別(5)"].map({1:"M",2:"F","1":"M","2":"F"})
    df = df.dropna(subset=["dx", "age", "sex"])

    # Keep sites with ≥ min_site_n patients (same as 01_build_matrix)
    site_counts = df.groupby("site")["pid"].nunique()
    keep = set(site_counts[site_counts >= min_site_n].index)
    df   = df[df["site"].isin(keep)]

    # Build vocabulary
    all_sites = sorted(keep)
    vocab     = {s: i + SITE_OFFSET for i, s in enumerate(all_sites)}
    pd.DataFrame({"site": all_sites,
                  "token_idx": [vocab[s] for s in all_sites]}).to_csv(
        DOUT / "transformer_site_vocab.csv", index=False)

    # Per-patient: earliest dx per site, ordered by date
    first = (df.sort_values("dx")
               .groupby(["pid","site"], as_index=False)
               .agg(dx=("dx","first"), age=("age","first"), sex=("sex","first")))
    first = first.sort_values(["pid","dx"])

    # Age / sex stats for normalisation
    age_mean = first["age"].mean(); age_std = first["age"].std() + 1e-6

    seqs = []
    for pid, grp in first.groupby("pid"):
        grp    = grp.sort_values("dx")
        t0     = grp["dx"].iloc[0]
        tokens = [vocab[s] for s in grp["site"]]
        days   = [(d - t0).days for d in grp["dx"]]
        ages   = [(a - age_mean) / age_std for a in grp["age"]]
        sex_b  = 1.0 if grp["sex"].iloc[0] == "M" else 0.0
        seqs.append({"pid": pid, "tokens": tokens, "days": days,
                     "ages": ages, "sex_bin": sex_b,
                     "n_cancers": len(tokens)})

    print(f"  {len(seqs):,} patients · {len(vocab)} sites · "
          f"vocab size {len(vocab) + SITE_OFFSET}")
    print(f"  Multi-cancer (≥2): {sum(1 for s in seqs if s['n_cancers']>=2):,}")
    return seqs, vocab


# ── Dataset ───────────────────────────────────────────────────────────────────

class CancerSeqDataset(Dataset):
    """
    Leave-one-out masking over MULTI-CANCER patients only.
    Each patient with k sites → k examples, one per masked site.
    Guarantees every example has exactly 1 masked position → no NaN loss.

    Single-cancer patients excluded: they produce no masked positions,
    causing cross_entropy(reduction='mean') to return NaN (0/0).
    """
    def __init__(self, seqs):
        self.examples = []
        for s in seqs:
            n = min(len(s["tokens"]), MAX_SEQ - 1)
            if n < 2:
                continue   # skip single-cancer patients
            for mask_pos in range(n):   # leave-one-out
                self.examples.append((s, mask_pos, n))

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        s, mask_pos, n = self.examples[idx]
        tokens = [CLS_IDX] + s["tokens"][:n]
        days   = [0.0]     + s["days"][:n]
        ages   = [s["ages"][0]] + s["ages"][:n]
        sex_b  = s["sex_bin"]
        labels = [-100] * len(tokens)

        # Mask exactly one site (mask_pos+1 to skip CLS at index 0)
        mp = mask_pos + 1
        labels[mp] = tokens[mp]
        tokens[mp] = MASK_IDX

        # Pad to MAX_SEQ
        pad_len = MAX_SEQ - len(tokens)
        tokens  += [PAD_IDX] * pad_len
        days    += [0.0]     * pad_len
        ages    += [0.0]     * pad_len
        labels  += [-100]    * pad_len
        attn     = [True] * (MAX_SEQ - pad_len) + [False] * pad_len

        return (torch.tensor(tokens,  dtype=torch.long),
                torch.tensor(days,    dtype=torch.float32),
                torch.tensor(ages,    dtype=torch.float32),
                torch.tensor(sex_b,   dtype=torch.float32),
                torch.tensor(attn,    dtype=torch.bool),
                torch.tensor(labels,  dtype=torch.long))


# ── Model ─────────────────────────────────────────────────────────────────────

class TimeEncoding(nn.Module):
    """Sinusoidal encoding on log(1 + days/30) — months scale."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, days):   # (B, S)
        t = torch.log1p(days / 30.0).unsqueeze(-1)           # (B, S, 1)
        div = torch.exp(torch.arange(0, self.d_model, 2,
                                      dtype=torch.float32,
                                      device=days.device)
                        * -(np.log(10000.0) / self.d_model))  # (d/2,)
        enc = torch.zeros(*days.shape, self.d_model, device=days.device)
        enc[..., 0::2] = torch.sin(t * div)
        enc[..., 1::2] = torch.cos(t * div[: self.d_model // 2])
        return enc


class CancerTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, ff_dim, dropout):
        super().__init__()
        self.site_embed  = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.time_enc    = TimeEncoding(d_model)
        self.age_proj    = nn.Linear(1, d_model)
        self.sex_embed   = nn.Embedding(2, d_model)
        self.input_norm  = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder     = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.mlm_head    = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size))
        self.d_model = d_model

    def forward(self, tokens, days, ages, sex_bin, attn_mask):
        # token embedding + time + age + sex
        x  = self.site_embed(tokens)                           # (B, S, D)
        x  = x + self.time_enc(days)                           # add time
        x  = x + self.age_proj(ages.unsqueeze(-1))             # add age
        x  = x + self.sex_embed(sex_bin.long()).unsqueeze(1)   # add sex (broadcast)
        x  = self.input_norm(x)
        # key_padding_mask: True = ignore (PAD positions)
        pad_mask = ~attn_mask                                  # (B, S) True=PAD
        h  = self.encoder(x, src_key_padding_mask=pad_mask)    # (B, S, D)
        logits = self.mlm_head(h)                              # (B, S, V)
        return logits, h[:, 0, :]                              # logits, CLS emb


def mlm_loss(logits, labels):
    """Cross-entropy only at masked positions. Safe when all labels=-100."""
    B, S, V = logits.shape
    mask = (labels != -100).view(-1)
    if mask.sum() == 0:
        return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()
    return nn.functional.cross_entropy(
        logits.view(B * S, V)[mask], labels.view(-1)[mask])


class _AllPatientsDataset(Dataset):
    """Unmasked forward pass for all patients — used only for CLS extraction."""
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        s      = self.seqs[idx]
        n      = min(len(s["tokens"]), MAX_SEQ - 1)
        tokens = [CLS_IDX] + s["tokens"][:n]
        days   = [0.0]     + s["days"][:n]
        ages   = [s["ages"][0]] + s["ages"][:n]
        sex_b  = s["sex_bin"]
        pad_len = MAX_SEQ - len(tokens)
        tokens  += [PAD_IDX] * pad_len
        days    += [0.0]     * pad_len
        ages    += [0.0]     * pad_len
        attn     = [True] * (MAX_SEQ - pad_len) + [False] * pad_len
        labels   = [-100] * MAX_SEQ
        return (torch.tensor(tokens, dtype=torch.long),
                torch.tensor(days,   dtype=torch.float32),
                torch.tensor(ages,   dtype=torch.float32),
                torch.tensor(sex_b,  dtype=torch.float32),
                torch.tensor(attn,   dtype=torch.bool),
                torch.tensor(labels, dtype=torch.long))


# ── Training ──────────────────────────────────────────────────────────────────

def recall_at_k(logits, labels, k_list=(1, 3, 5)):
    mask = labels != -100
    if mask.sum() == 0: return {f"R@{k}": 0.0 for k in k_list}
    pred  = logits[mask]                          # (M, V)
    true  = labels[mask]                          # (M,)
    ranks = torch.argsort(pred, dim=1, descending=True)
    res   = {}
    for k in k_list:
        top_k = ranks[:, :k]
        hits  = (top_k == true.unsqueeze(1)).any(dim=1)
        res[f"R@{k}"] = hits.float().mean().item()
    return res


def main():
    print("=== Registry DL — 07: Cancer Sequence Transformer ===")
    seqs, vocab = build_sequences()
    vocab_size  = len(vocab) + SITE_OFFSET   # PAD + MASK + CLS + sites

    # Train / val split at PATIENT level (leave-one-out within each patient)
    # Only multi-cancer patients contribute training examples
    multi_seqs = [s for s in seqs if s["n_cancers"] >= 2]
    rng        = np.random.default_rng(SEED)
    idx        = rng.permutation(len(multi_seqs))
    n_val      = int(len(multi_seqs) * VAL_FRAC)
    val_seqs   = [multi_seqs[i] for i in idx[:n_val]]
    train_seqs = [multi_seqs[i] for i in idx[n_val:]]
    print(f"  Multi-cancer train/val: {len(train_seqs):,} / {len(val_seqs):,} patients")

    ds_tr  = CancerSeqDataset(train_seqs)
    ds_val = CancerSeqDataset(val_seqs)
    dl_tr  = DataLoader(ds_tr,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CancerTransformer(vocab_size, D_MODEL, N_HEAD, N_LAYERS, FF_DIM, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    history = []
    best_val, patience_ctr = float("inf"), 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for tokens, days, ages, sex_b, attn, labels in dl_tr:
            opt.zero_grad()
            logits, _ = model(tokens, days, ages, sex_b, attn)
            loss = mlm_loss(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        sched.step()

        model.eval(); vl_loss = 0.0
        with torch.no_grad():
            for tokens, days, ages, sex_b, attn, labels in dl_val:
                logits, _ = model(tokens, days, ages, sex_b, attn)
                vl_loss  += mlm_loss(logits, labels).item()

        tr_loss /= len(dl_tr); vl_loss /= len(dl_val)
        history.append({"epoch": epoch, "train": tr_loss, "val": vl_loss})

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={tr_loss:.4f} val={vl_loss:.4f}")

        if vl_loss < best_val - 1e-5:
            best_val, patience_ctr = vl_loss, 0
            torch.save({"model": model.state_dict(), "vocab": vocab,
                        "vocab_size": vocab_size},
                       MOUT / "transformer_weights.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop epoch {epoch}"); break

    # ── Evaluation ────────────────────────────────────────────────────────────
    ckpt  = torch.load(MOUT / "transformer_weights.pt")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # R@k on val masked positions
    all_logits, all_labels = [], []
    with torch.no_grad():
        for tokens, days, ages, sex_b, attn, labels in dl_val:
            logits, _ = model(tokens, days, ages, sex_b, attn)
            all_logits.append(logits); all_labels.append(labels)
    all_logits = torch.cat(all_logits); all_labels = torch.cat(all_labels)
    rec = recall_at_k(all_logits, all_labels)
    print(f"\n  Validation Recall@k:")
    for k, v in rec.items(): print(f"    {k}: {v:.3f}")

    # Patient CLS embeddings — all patients (single + multi) via unmasked pass
    print("  Extracting CLS embeddings (all patients)…")
    all_seqs_ds = CancerSeqDataset(seqs)   # only multi-cancer; single handled below
    # Build a simple unmasked dataset for ALL patients for CLS extraction
    all_seqs_ds = _AllPatientsDataset(seqs)
    all_dl      = DataLoader(all_seqs_ds, batch_size=512, shuffle=False)
    cls_list    = []
    with torch.no_grad():
        for tokens, days, ages, sex_b, attn, _ in all_dl:
            _, cls = model(tokens, days, ages, sex_b, attn)
            cls_list.append(cls.numpy())
    cls_emb = np.concatenate(cls_list, axis=0)
    np.save(DOUT / "patient_cls_embeddings.npy", cls_emb)
    print(f"  CLS embeddings: {cls_emb.shape}")

    # Val predictions CSV (masked positions only)
    inv_vocab  = {v: k for k, v in vocab.items()}
    pred_rows  = []
    mask_flat  = all_labels.view(-1) != -100
    logits_flat = all_logits.view(-1, all_logits.size(-1))[mask_flat]
    labels_flat = all_labels.view(-1)[mask_flat]
    ranks_flat  = torch.argsort(logits_flat, dim=1, descending=True)
    for i in range(len(labels_flat)):
        true_tok  = labels_flat[i].item()
        rank_true = int((ranks_flat[i] == true_tok).nonzero()[0].item()) + 1
        top3      = [inv_vocab.get(ranks_flat[i, j].item(), f"tok{ranks_flat[i,j].item()}")
                     for j in range(3)]
        pred_rows.append({"true_site": inv_vocab.get(true_tok, f"tok{true_tok}"),
                          "rank_of_true": rank_true, "top3": ", ".join(top3)})
    pd.DataFrame(pred_rows).to_csv(OUT / "val_predictions.csv", index=False)

    # Figures
    hist_df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist_df["epoch"], hist_df["train"], label="train")
    ax.plot(hist_df["epoch"], hist_df["val"],   label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MLM cross-entropy")
    ax.set_title(f"Cancer Transformer — masked site prediction\n"
                 f"({n_params:,} params, d={D_MODEL}, L={N_LAYERS}, H={N_HEAD})")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig_train_loss.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    ks = list(rec.keys()); vs = [rec[k] for k in ks]
    bars = ax.bar(ks, vs, color=["#14304a","#2e7fbf","#7eb8df"])
    ax.set_ylim(0, 1); ax.set_ylabel("Recall")
    ax.set_title("Transformer next-cancer accuracy\n(val masked positions)")
    for b, v in zip(bars, vs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_recall_at_k.png", dpi=150); plt.close()

    print(f"\n  Best val loss: {best_val:.5f}")
    print(f"  Figures → results/07_transformer/")


if __name__ == "__main__":
    main()
