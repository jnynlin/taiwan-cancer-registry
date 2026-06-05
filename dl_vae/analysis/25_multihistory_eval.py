"""
Registry DL — Script 25: Multi-History Transformer Evaluation

The surveillance calendar (Script 12) used only the FIRST cancer as context:
  [CLS, c1, MASK]  →  R@1=0.232  (8.6× random)

The trained Transformer achieves R@1=0.312 with full history (Script 08).
The gap 0.232 → 0.312 represents the value of temporal sequence context.

This script evaluates the Transformer with variable context lengths:
  k=1 context: [CLS, c1,     MASK] — first cancer only
  k=2 context: [CLS, c1, c2, MASK] — first two cancers (requires ≥3-cancer patients)
  k≥3 context: [CLS, c1, c2, c3, MASK] — three or more (requires ≥4-cancer patients)

For each context length, computes R@1/R@3/R@5 and probability calibration.
Generates an updated surveillance calendar using the longest available context.

Outputs:
  results/25_multihistory/accuracy_by_context.csv
  results/25_multihistory/multihistory_calendar.csv
  results/25_multihistory/fig_accuracy_by_context.png
  results/25_multihistory/fig_prob_calibration.png
  results/25_multihistory/fig_context_value.png
  results/25_multihistory/fig_site_improvement.png
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE  = Path(__file__).parent.parent
RAW   = BASE.parent / "data/processed/all_cancers.csv"
CKPT  = BASE / "models/transformer_weights.pt"
DOUT  = BASE / "data"
R12   = BASE / "results/12_surveillance"
OUT   = BASE / "results/25_multihistory"
OUT.mkdir(parents=True, exist_ok=True)

# ── Locked constants (identical to Script 07) ─────────────────────────────────
MAX_SEQ    = 8
PAD_IDX    = 0
MASK_IDX   = 1
CLS_IDX    = 2
SITE_OFFSET = 3
SEED       = 42
VAL_FRAC   = 0.20
D_MODEL    = 64
N_HEAD     = 4
N_LAYERS   = 2
FF_DIM     = 128      # actual value from checkpoint (128, not 256)
DROPOUT    = 0.1
# ─────────────────────────────────────────────────────────────────────────────


# ── Model (exact copy from Script 07) ────────────────────────────────────────
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
    def __init__(self, vocab_size, d_model, nhead, num_layers, ff_dim, dropout):
        super().__init__()
        self.site_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.age_proj   = nn.Linear(1, d_model)
        self.sex_embed  = nn.Embedding(2, d_model)
        self.time_enc   = TimeEncoding(d_model)
        self.input_norm = nn.LayerNorm(d_model)
        enc_layer       = nn.TransformerEncoderLayer(d_model, nhead, ff_dim,
                                                     dropout, batch_first=True,
                                                     norm_first=True)
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.mlm_head   = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size))

    def forward(self, tokens, days, ages, sex_bin, attn_mask):
        x = self.site_embed(tokens)
        x = x + self.time_enc(days)
        x = x + self.age_proj(ages.unsqueeze(-1))
        x = x + self.sex_embed(sex_bin.long()).unsqueeze(1)   # sex is (B,) → broadcast
        x = self.input_norm(x)
        h = self.encoder(x, src_key_padding_mask=~attn_mask)
        return self.mlm_head(h), h[:, 0, :]
# ─────────────────────────────────────────────────────────────────────────────


def roc_to_ts(x):
    s = str(x).split(".")[0].zfill(7)
    if not s.isdigit() or len(s) != 7: return pd.NaT
    y  = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00","99") else "01"
    dd = s[5:7] if s[5:7] not in ("00","99") else "01"
    try: return pd.Timestamp(f"{y}-{mm}-{dd}")
    except: return pd.NaT


def build_val_sequences():
    """Reconstruct val sequences with dates. Same logic + SEED as Scripts 07/12."""
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = df.columns.str.strip().str.lstrip("﻿")
    df["pid"]   = df["病歷號(2)"].astype(str).str.strip()
    df["site"]  = df["腫瘤部位(47)"].astype(str).str[:3].str.upper()
    df["age"]   = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["sex"]   = df["性別(5)"].map({1:"M",2:"F","1":"M","2":"F"})
    df["dx_ts"] = df["最初診斷日(45)"].apply(roc_to_ts)
    df = df.dropna(subset=["dx_ts","age","sex"])

    vocab_df  = pd.read_csv(DOUT / "transformer_site_vocab.csv")
    vocab     = {row["site"]: row["token_idx"] for _, row in vocab_df.iterrows()}
    inv_vocab = {v: k for k, v in vocab.items()}
    df = df[df["site"].isin(vocab.keys())]

    # Bug fix 1: deduplicate per (pid, site) — same as Scripts 07/12
    first = (df.sort_values("dx_ts")
               .groupby(["pid","site"], as_index=False)
               .agg(dx_ts=("dx_ts","first"), age=("age","first"), sex=("sex","first")))
    first = first.sort_values(["pid","dx_ts"])

    # Bug fix 2: normalize ages using checkpoint's saved age_mean/age_std
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    age_mean = ck.get("age_mean", first["age"].mean())
    age_std  = ck.get("age_std",  first["age"].std() + 1e-6)

    seqs = []
    for pid, grp in first.groupby("pid"):
        grp    = grp.sort_values("dx_ts")
        tokens = [vocab[s] for s in grp["site"]]
        ages   = [(a - age_mean) / age_std for a in grp["age"]]
        sex_b  = 1 if grp["sex"].iloc[0] == "M" else 0
        dates  = grp["dx_ts"].tolist()
        sites  = grp["site"].tolist()
        seqs.append({"pid": pid, "tokens": tokens, "ages": ages,
                     "sex": sex_b, "dates": dates, "sites": sites,
                     "n_cancers": len(tokens)})

    multi_seqs = [s for s in seqs if s["n_cancers"] >= 2]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(multi_seqs))
    n_val = int(len(multi_seqs) * VAL_FRAC)
    val_seqs = [multi_seqs[i] for i in idx[:n_val]]
    return val_seqs, vocab, inv_vocab


def predict_with_context(model, context_tokens, context_ages, sex_b, inv_vocab,
                         mask_pos):
    """
    Run inference with a given context prefix.

    context_tokens : list of k site token indices (already seen cancers)
    mask_pos       : position in the sequence where MASK is placed (= k+1)
                     Sequence layout: [CLS, c1, ..., ck, MASK, PAD, ..., PAD]
    """
    k = len(context_tokens)
    assert mask_pos == k + 1, "mask_pos must be immediately after context"

    seq_len = MAX_SEQ
    n_pad   = seq_len - k - 2
    tokens  = [CLS_IDX] + context_tokens + [MASK_IDX] + [PAD_IDX] * n_pad
    days    = [0.0] * seq_len
    ages    = [context_ages[0]] + list(context_ages) + [0.0] * (n_pad + 1)  # +1 for MASK
    attn    = [True] * (k + 2) + [False] * n_pad

    t = torch.tensor([tokens], dtype=torch.long)
    d = torch.tensor([days],   dtype=torch.float)
    a = torch.tensor([ages],   dtype=torch.float)
    s = torch.tensor([sex_b],  dtype=torch.float)   # (B,) — sex scalar per sample
    m = torch.tensor([attn],   dtype=torch.bool)

    with torch.no_grad():
        logits, _ = model(t, d, a, s, m)
        probs = torch.softmax(logits[0, mask_pos, :], dim=-1).clone()

    # Suppress special tokens and all already-seen context tokens
    for bad in [PAD_IDX, MASK_IDX, CLS_IDX] + context_tokens:
        probs[bad] = 0.0

    top5 = torch.argsort(probs, descending=True)[:5]
    return [(inv_vocab[tok.item()], float(probs[tok].item()))
            for tok in top5 if tok.item() in inv_vocab]


def main():
    print("=== Registry DL — 25: Multi-History Transformer Evaluation ===")

    # ── Load model ────────────────────────────────────────────────────────────
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    vocab_size = len(ck["vocab"]) + SITE_OFFSET
    model = CancerTransformer(vocab_size, D_MODEL, N_HEAD, N_LAYERS, FF_DIM, DROPOUT)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"  Model loaded: vocab_size={vocab_size}, {sum(p.numel() for p in model.parameters()):,} params")

    # ── Rebuild val sequences ─────────────────────────────────────────────────
    val_seqs, vocab, inv_vocab = build_val_sequences()
    print(f"  Val patients: {len(val_seqs):,}")
    n_by_length = pd.Series([s["n_cancers"] for s in val_seqs]).value_counts().sort_index()
    print(f"  Cancer count distribution:\n{n_by_length.to_string()}")

    # ── Evaluate by context length ────────────────────────────────────────────
    rows = []
    for seq in val_seqs:
        n = min(seq["n_cancers"], MAX_SEQ)  # cap at MAX_SEQ-1 context positions
        for k in range(1, n):               # k = n cancers in context, predict k+1
            if k + 2 > MAX_SEQ: break       # need room for [CLS]+context+[MASK]
            context_tokens = seq["tokens"][:k]
            context_ages   = seq["ages"][:k]
            target_token   = seq["tokens"][k]
            target_site    = inv_vocab.get(target_token, None)
            if target_site is None: continue

            preds = predict_with_context(
                model, context_tokens, context_ages, seq["sex"], inv_vocab,
                mask_pos=k + 1)

            if not preds: continue
            pred_sites = [p[0] for p in preds]
            prob1      = preds[0][1]
            rows.append({
                "pid":           seq["pid"],
                "context_len":   k,
                "context_sites": "+".join(inv_vocab.get(t, "?") for t in context_tokens),
                "target_site":   target_site,
                "pred1":         pred_sites[0] if len(pred_sites) > 0 else None,
                "pred2":         pred_sites[1] if len(pred_sites) > 1 else None,
                "pred3":         pred_sites[2] if len(pred_sites) > 2 else None,
                "prob1":         round(prob1, 4),
                "hit@1":         int(pred_sites[0] == target_site) if pred_sites else 0,
                "hit@3":         int(target_site in pred_sites[:3]),
                "hit@5":         int(target_site in pred_sites[:5]),
            })

    eval_df = pd.DataFrame(rows)
    print(f"\n  Total inference rows: {len(eval_df):,}")

    # ── Accuracy by context length ────────────────────────────────────────────
    acc = (eval_df.groupby("context_len")
           .agg(n=("hit@1","count"),
                R_at_1=("hit@1","mean"),
                R_at_3=("hit@3","mean"),
                R_at_5=("hit@5","mean"),
                mean_prob1=("prob1","mean"))
           .reset_index())
    acc.to_csv(OUT / "accuracy_by_context.csv", index=False)

    print("\n  Accuracy by context length:")
    print(acc.to_string(index=False))

    # Baseline: Script 12 R@1=0.232 (first-cancer only, same context_len=1)
    script12_r1 = 0.232

    # ── Fig A: R@k bar chart by context length ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = acc["context_len"].values
    labels = [f"k={int(k)}\nn={int(n)}" for k, n in zip(acc["context_len"], acc["n"])]

    for ax, col, title in [
        (axes[0], "R_at_1", "R@1 (Top-1 accuracy)"),
        (axes[1], "R_at_3", "R@3 (Top-3 accuracy)"),
        (axes[2], "R_at_5", "R@5 (Top-5 accuracy)"),
    ]:
        bars = ax.bar(range(len(x)), acc[col].values, color="#2e7fbf", alpha=0.8)
        ax.axhline(script12_r1 if col=="R_at_1" else (0.498 if col=="R_at_3" else 0.664),
                   color="orange", lw=1.5, ls="--",
                   label=f"Script 12 baseline ({script12_r1 if col=='R_at_1' else (0.498 if col=='R_at_3' else 0.664):.3f})")
        ax.axhline(1/37, color="gray", lw=1, ls=":", label="Random (1/37)")
        ax.set_xticks(range(len(x))); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_ylim(0, min(1, acc[col].max() * 1.3))
        ax.legend(fontsize=8)
        for bar, val in zip(bars, acc[col].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("Transformer accuracy by context length\n"
                 "(k=context length in cancers; does more history → better prediction?)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_accuracy_by_context.png", dpi=150)
    plt.close()

    # ── Fig B: Context value — marginal lift ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    r1_vals = acc["R_at_1"].values
    context_lens = acc["context_len"].values
    n_vals = acc["n"].values

    ax.plot(context_lens, r1_vals, marker="o", ms=8, color="#2e7fbf", lw=2,
            label="R@1 by context length")
    ax.axhline(script12_r1, color="orange", lw=1.5, ls="--",
               label=f"Script 12 (first-only) R@1={script12_r1:.3f}")
    ax.axhline(0.312, color="green", lw=1.5, ls="--",
               label="Script 08 (full history) R@1=0.312")
    ax.axhline(1/37, color="gray", lw=1, ls=":", label="Random baseline (1/37=0.027)")
    for k, r, n in zip(context_lens, r1_vals, n_vals):
        ax.annotate(f"{r:.3f}\n(n={n})", xy=(k, r), xytext=(k+0.05, r+0.01),
                    fontsize=8, ha="left")
    ax.set_xlabel("Context length k (number of cancers used as input)")
    ax.set_ylabel("R@1")
    ax.set_title("Incremental value of cancer history for next-cancer prediction\n"
                 "Does each additional cancer in context improve prediction accuracy?")
    ax.legend(fontsize=9)
    ax.set_xlim(0.5, max(context_lens)+0.8)
    ax.set_ylim(0, 0.45)
    fig.tight_layout()
    fig.savefig(OUT / "fig_context_value.png", dpi=150)
    plt.close()

    # ── Fig C: Calibration — prob1 vs hit@1 by context length ─────────────────
    fig, axes = plt.subplots(1, min(3, len(acc)), figsize=(14, 5))
    if not hasattr(axes, '__len__'): axes = [axes]
    for ax, (_, row) in zip(axes, acc[acc["context_len"] <= 3].iterrows()):
        k = int(row["context_len"])
        sub = eval_df[eval_df["context_len"] == k].copy()
        sub["prob1_bin"] = pd.cut(sub["prob1"], bins=10)
        cal = sub.groupby("prob1_bin").agg(
            mean_prob=("prob1","mean"), hit_rate=("hit@1","mean"), n=("hit@1","count")
        ).reset_index()
        cal = cal[cal["n"] >= 3]
        ax.scatter(cal["mean_prob"], cal["hit_rate"], s=cal["n"]*3,
                   alpha=0.7, color="#2e7fbf", edgecolors="white")
        ax.plot([0,1],[0,1], "k--", lw=1, label="Perfect calibration")
        rho, p = stats.spearmanr(cal["mean_prob"], cal["hit_rate"])
        ax.set_title(f"Context k={k} (n={int(row['n'])})\nρ={rho:.2f} p={p:.3f}", fontsize=9)
        ax.set_xlabel("Model prob1"); ax.set_ylabel("Empirical hit@1")
        ax.set_xlim(0, 0.8); ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle("Probability calibration by context length\n"
                 "(bubble size = n patients in bin)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_prob_calibration.png", dpi=150)
    plt.close()

    # ── Fig D: Per-site R@1 improvement k=1 vs k=2 ───────────────────────────
    if 2 in eval_df["context_len"].values:
        k1_site = (eval_df[eval_df["context_len"]==1]
                   .groupby("target_site").agg(r1_k1=("hit@1","mean"),
                                               n_k1=("hit@1","count")).reset_index())
        k2_site = (eval_df[eval_df["context_len"]==2]
                   .groupby("target_site").agg(r1_k2=("hit@1","mean"),
                                               n_k2=("hit@1","count")).reset_index())
        site_comp = k1_site.merge(k2_site, on="target_site", how="inner")
        site_comp = site_comp[(site_comp["n_k1"]>=10) & (site_comp["n_k2"]>=5)]
        site_comp["delta_r1"] = site_comp["r1_k2"] - site_comp["r1_k1"]
        site_comp = site_comp.sort_values("delta_r1", ascending=False)

        if len(site_comp) > 0:
            fig, ax = plt.subplots(figsize=(9, max(5, len(site_comp)*0.4+1.5)))
            colors = ["#2ca02c" if d >= 0 else "#d62728" for d in site_comp["delta_r1"]]
            ax.barh(site_comp["target_site"], site_comp["delta_r1"],
                    color=colors, alpha=0.8)
            ax.axvline(0, color="black", lw=1)
            ax.set_xlabel("ΔR@1 (k=2 context − k=1 context)")
            ax.set_title("Per-site R@1 improvement: 2-cancer context vs 1-cancer context\n"
                         "(green = improves; red = degrades; sites with n≥10/k=1, n≥5/k=2)")
            fig.tight_layout()
            fig.savefig(OUT / "fig_site_improvement.png", dpi=150)
            plt.close()

    # ── Build updated multi-history surveillance calendar ─────────────────────
    # For each val patient: use the longest available context
    print("\n  Building multi-history surveillance calendar...")
    cal_rows = []
    for seq in val_seqs:
        n = min(seq["n_cancers"], MAX_SEQ)
        for k in range(1, n):
            if k + 2 > MAX_SEQ: break
            context_tokens = seq["tokens"][:k]
            context_ages   = seq["ages"][:k]
            target_token   = seq["tokens"][k]
            target_site    = inv_vocab.get(target_token)
            if target_site is None: continue

            preds = predict_with_context(
                model, context_tokens, context_ages, seq["sex"], inv_vocab,
                mask_pos=k + 1)
            if not preds: continue

            # Compute actual gap (days between cancer k and k+1)
            actual_gap = None
            if len(seq["dates"]) > k and seq["dates"][k-1] is not None and seq["dates"][k] is not None:
                actual_gap = (seq["dates"][k] - seq["dates"][k-1]).days

            cal_rows.append({
                "pid":           seq["pid"],
                "context_len":   k,
                "context_sites": "+".join(inv_vocab.get(t, "?") for t in context_tokens),
                "target_site":   target_site,
                "actual_gap_days": actual_gap,
                "pred1": preds[0][0] if len(preds)>0 else None,
                "prob1": round(preds[0][1],4) if len(preds)>0 else None,
                "pred2": preds[1][0] if len(preds)>1 else None,
                "pred3": preds[2][0] if len(preds)>2 else None,
                "hit@1": int(preds[0][0]==target_site) if preds else 0,
                "hit@3": int(target_site in [p[0] for p in preds[:3]]),
                "hit@5": int(target_site in [p[0] for p in preds[:5]]),
            })

    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(OUT / "multihistory_calendar.csv", index=False)
    print(f"  Calendar: {len(cal_df):,} rows, {cal_df['pid'].nunique():,} unique patients")

    # ── Summary print ─────────────────────────────────────────────────────────
    print("\n  === SUMMARY ===")
    print(f"  Script 12 baseline (k=1 only):  R@1={script12_r1:.3f}")
    for _, row in acc.iterrows():
        print(f"  This script (k={int(row['context_len'])}, n={int(row['n']):,}): "
              f"R@1={row['R_at_1']:.3f}  R@3={row['R_at_3']:.3f}  R@5={row['R_at_5']:.3f}")
    print(f"  Script 08 full-history ceiling: R@1=0.312")
    print(f"\n  Saved → {OUT}/")


if __name__ == "__main__":
    main()
