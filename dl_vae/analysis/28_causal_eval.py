"""
Registry DL — Script 28: Causal vs BERT Multi-History Comparison

Loads both checkpoints and evaluates R@k at all context lengths k=1..4.
Key question: does causal retraining improve R@1 with longer history?

BERT (Script 07): R@1 stays flat or drops at k>1 (can't use temporal order)
Causal (Script 27): R@1 should increase with k (uses order correctly)

Outputs:
  results/28_causal_eval/accuracy_comparison.csv
  results/28_causal_eval/fig_causal_vs_bert.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

D_MODEL = 64; N_HEAD = 4; N_LAYERS = 2; FF_DIM = 128
MAX_SEQ = 8; DROPOUT = 0.1
PAD_IDX = 0; MASK_IDX = 1; CLS_IDX = 2; SITE_OFFSET = 3
VAL_FRAC = 0.20; SEED = 42

BASE  = Path(__file__).parent.parent
RAW   = BASE.parent / "data/processed/all_cancers.csv"
MOUT  = BASE / "models"
OUT   = BASE / "results/28_causal_eval"
OUT.mkdir(parents=True, exist_ok=True)


# ── Model (identical to Scripts 07 / 27) ──────────────────────────────────────

def roc_to_ts(x):
    s = str(x).split(".")[0].zfill(7)
    if not s.isdigit() or len(s) != 7: return pd.NaT
    y = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00","99") else "01"
    dd = s[5:7] if s[5:7] not in ("00","99") else "01"
    try: return pd.Timestamp(f"{y}-{mm}-{dd}")
    except: return pd.NaT


class TimeEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    def forward(self, days):
        t   = torch.log1p(days / 30.0).unsqueeze(-1)
        div = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32,
                                      device=days.device)
                        * -(np.log(10000.0) / self.d_model))
        enc = torch.zeros(*days.shape, self.d_model, device=days.device)
        enc[..., 0::2] = torch.sin(t * div)
        enc[..., 1::2] = torch.cos(t * div[:self.d_model // 2])
        return enc


class CancerTransformer(nn.Module):
    def __init__(self, vocab_size, causal=False):
        super().__init__()
        self.causal     = causal
        self.site_embed = nn.Embedding(vocab_size, D_MODEL, padding_idx=PAD_IDX)
        self.time_enc   = TimeEncoding(D_MODEL)
        self.age_proj   = nn.Linear(1, D_MODEL)
        self.sex_embed  = nn.Embedding(2, D_MODEL)
        self.input_norm = nn.LayerNorm(D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True, norm_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.mlm_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL), nn.GELU(),
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, vocab_size))

    def forward(self, tokens, days, ages, sex_bin, attn_mask):
        x = self.site_embed(tokens)
        x = x + self.time_enc(days)
        x = x + self.age_proj(ages.unsqueeze(-1))
        x = x + self.sex_embed(sex_bin.long()).unsqueeze(1)
        x = self.input_norm(x)
        pad_mask = ~attn_mask
        causal_mask = None
        if self.causal:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                tokens.size(1), device=tokens.device)
        h = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        return self.mlm_head(h), h[:, 0, :]


# ── Val sequences (identical split to Scripts 07 / 25) ────────────────────────

def build_val_seqs(vocab, age_mean, age_std, min_site_n=30):
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = df.columns.str.strip().str.lstrip('﻿')
    df["pid"]  = df["病歷號(2)"].astype(str).str.strip()
    df["site"] = df["腫瘤部位(47)"].astype(str).str[:3].str.upper()
    df["dx"]   = df["最初診斷日(45)"].apply(roc_to_ts)
    df["age"]  = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["sex"]  = df["性別(5)"].map({1:"M",2:"F","1":"M","2":"F"})
    df = df.dropna(subset=["dx","age","sex"])
    site_counts = df.groupby("site")["pid"].nunique()
    keep = set(site_counts[site_counts >= min_site_n].index)
    df   = df[df["site"].isin(keep)]

    first = (df.sort_values("dx")
               .groupby(["pid","site"], as_index=False)
               .agg(dx=("dx","first"), age=("age","first"), sex=("sex","first")))
    first = first.sort_values(["pid","dx"])

    seqs = []
    for pid, grp in first.groupby("pid"):
        grp    = grp.sort_values("dx")
        t0     = grp["dx"].iloc[0]
        tokens = [vocab.get(s) for s in grp["site"]]
        tokens = [t for t in tokens if t is not None]
        if not tokens: continue
        days   = [(d - t0).days for d in grp["dx"]][:len(tokens)]
        ages   = [(a - age_mean) / age_std for a in grp["age"]][:len(tokens)]
        sex_b  = 1.0 if grp["sex"].iloc[0] == "M" else 0.0
        seqs.append({"pid": pid, "tokens": tokens, "days": days,
                     "ages": ages, "sex_bin": sex_b,
                     "n_cancers": len(tokens)})

    multi = [s for s in seqs if s["n_cancers"] >= 2]
    rng   = np.random.default_rng(SEED)
    idx   = rng.permutation(len(multi))
    n_val = int(len(multi) * VAL_FRAC)
    return [multi[i] for i in idx[:n_val]]


# ── Per-k evaluation ───────────────────────────────────────────────────────────

def eval_at_k(model, val_seqs, inv_vocab, device, k_ctx):
    """R@1, R@3, R@5 at a fixed context length k_ctx."""
    model.eval()
    hits1 = hits3 = hits5 = total = 0
    with torch.no_grad():
        for s in val_seqs:
            if s["n_cancers"] <= k_ctx:
                continue
            ctx = s["tokens"][:k_ctx]
            tgt = inv_vocab.get(s["tokens"][k_ctx])
            if tgt is None:
                continue

            toks = [CLS_IDX] + ctx + [MASK_IDX]
            d2   = [0.0] + s["days"][:k_ctx] + [float(s["days"][k_ctx-1]) + 1]
            a2   = [s["ages"][0]] + s["ages"][:k_ctx] + [0.0]
            sl   = len(toks)
            pl   = MAX_SEQ - sl
            toks += [PAD_IDX]*pl; d2 += [0.0]*pl; a2 += [0.0]*pl
            atn  = [True]*sl + [False]*pl
            mp   = k_ctx + 1

            t2  = torch.tensor([toks], dtype=torch.long,    device=device)
            dd  = torch.tensor([d2],   dtype=torch.float32, device=device)
            aa  = torch.tensor([a2],   dtype=torch.float32, device=device)
            sx  = torch.tensor([s["sex_bin"]], dtype=torch.float32, device=device)
            mm  = torch.tensor([atn],  dtype=torch.bool,    device=device)

            logits, _ = model(t2, dd, aa, sx, mm)
            probs = torch.softmax(logits[0, mp, :], dim=-1).clone()
            for bad in [PAD_IDX, MASK_IDX, CLS_IDX] + ctx:
                if bad < probs.size(0): probs[bad] = 0.0

            top5 = [inv_vocab.get(i.item()) for i in torch.argsort(probs, descending=True)[:5]]
            total += 1
            if tgt == top5[0]:           hits1 += 1
            if tgt in top5[:3]:          hits3 += 1
            if tgt in top5[:5]:          hits5 += 1

    if total == 0:
        return None
    return {"k": k_ctx, "n": total,
            "R_at_1": hits1/total, "R_at_3": hits3/total, "R_at_5": hits5/total}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bert_path   = MOUT / "transformer_weights.pt"
    causal_path = MOUT / "causal_transformer_weights.pt"

    missing = [p for p in (bert_path, causal_path) if not p.exists()]
    if missing:
        print("Missing checkpoints:", [str(p) for p in missing])
        print("Run Script 07 (BERT) and Script 27 (Causal) first.")
        return

    # Load BERT checkpoint
    bert_ckpt  = torch.load(bert_path,   map_location=device, weights_only=False)
    caus_ckpt  = torch.load(causal_path, map_location=device, weights_only=False)

    vocab      = bert_ckpt["vocab"]
    vocab_size = bert_ckpt["vocab_size"]
    # BERT checkpoint predates age_mean/age_std saving; fall back to causal ckpt
    age_mean   = bert_ckpt.get("age_mean", caus_ckpt["age_mean"])
    age_std    = bert_ckpt.get("age_std",  caus_ckpt["age_std"])
    inv_vocab  = {v: k for k, v in vocab.items()}

    print("Loading BERT model…")
    bert_model = CancerTransformer(vocab_size, causal=False).to(device)
    bert_model.load_state_dict(bert_ckpt["model"])

    print("Loading Causal model…")
    caus_model = CancerTransformer(vocab_size, causal=True).to(device)
    caus_model.load_state_dict(caus_ckpt["model"])

    print("Building val sequences…")
    val_seqs = build_val_seqs(vocab, age_mean, age_std)
    random_r1 = 1 / len(vocab)
    print(f"  Val patients (≥2 cancers): {len(val_seqs):,}")
    print(f"  Random R@1 baseline: {random_r1:.4f}")

    # Evaluate both models at k=1..4
    rows = []
    print("\nContext k | BERT R@1 | Causal R@1 | n")
    print("-" * 45)
    for k in range(1, MAX_SEQ - 2):
        b = eval_at_k(bert_model,  val_seqs, inv_vocab, device, k)
        c = eval_at_k(caus_model,  val_seqs, inv_vocab, device, k)
        if b is None or c is None or b["n"] < 5:
            break
        print(f"  k={k}     {b['R_at_1']:.3f}      {c['R_at_1']:.3f}        n={b['n']}")
        rows.append({
            "context_k": k, "n": b["n"],
            "bert_r1":   round(b["R_at_1"], 4),
            "bert_r3":   round(b["R_at_3"], 4),
            "bert_r5":   round(b["R_at_5"], 4),
            "causal_r1": round(c["R_at_1"], 4),
            "causal_r3": round(c["R_at_3"], 4),
            "causal_r5": round(c["R_at_5"], 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "accuracy_comparison.csv", index=False)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ks = df["context_k"].tolist()

    for ax, metric, label in zip(axes, ["r1","r3","r5"], ["R@1","R@3","R@5"]):
        ax.plot(ks, df[f"bert_r{metric[-1]}"],   "o-", color="#2563eb",
                label="BERT (Script 07)", linewidth=2)
        ax.plot(ks, df[f"causal_r{metric[-1]}"], "s-", color="#16a34a",
                label="Causal (Script 27)", linewidth=2)
        ax.axhline(random_r1, color="#9ca3af", linestyle="--",
                   label=f"Random ({random_r1:.3f})", linewidth=1)
        ax.set_xlabel("Context length k")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by Cancer History Length")
        ax.set_xticks(ks)
        ax.set_xticklabels([f"k={k}\n(n={row['n']})"
                            for k, row in zip(ks, rows)], fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("CMUH Registry — BERT vs Causal Transformer\n"
                 "Causal model should improve with k; BERT should not",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_causal_vs_bert.png", dpi=150)
    plt.close(fig)
    print(f"\nOutputs → {OUT}/")


if __name__ == "__main__":
    main()
