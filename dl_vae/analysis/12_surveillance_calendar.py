"""
Registry DL — Script 12: Sequence-Aware Surveillance Calendar

Clinical framing: a patient presents with cancer at site A on date T0.
The Transformer is queried with [CLS, A, MASK] to predict the most likely
next cancer site. The MASK position is placed at day 0 (unknown future time);
probability reflects the model's site preference given A as context.

Timing windows are derived empirically from training-set transition gaps
(actual day differences between first and second cancers per site pair).

Validation: 20% held-out multi-cancer patients (same SEED=42 split as Script 07).
For each validation patient, the model sees only the first cancer and predicts
the next; we compare to the actual second cancer site and record the gap.

Outputs:
  results/12_surveillance/surveillance_calendar.csv
    pid, first_site, true_second_site, actual_gap_days,
    pred1..pred5, prob1..prob5, hit@1, hit@3, hit@5,
    timing_p25, timing_median, timing_p75 (days, from empirical training pairs)
  results/12_surveillance/timing_windows.csv
    first_site, second_site, n_pairs, gap_p25, gap_median, gap_p75
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: pip install torch"); raise SystemExit(1)

BASE    = Path(__file__).parent.parent
RAW     = BASE.parent / "data/processed/all_cancers.csv"
DOUT    = BASE / "data"
MOUT    = BASE / "models"
OUT     = BASE / "results/12_surveillance"
OUT.mkdir(parents=True, exist_ok=True)

# ── Copied constants / classes from Script 07 (cannot import directly) ────────
D_MODEL     = 64
N_HEAD      = 4
N_LAYERS    = 2
FF_DIM      = 128
MAX_SEQ     = 8
DROPOUT     = 0.1
VAL_FRAC    = 0.20
SEED        = 42
PAD_IDX     = 0
MASK_IDX    = 1
CLS_IDX     = 2
SITE_OFFSET = 3


def roc_to_ts(x):
    s = str(x).split(".")[0].zfill(7)
    if not s.isdigit() or len(s) != 7: return pd.NaT
    y  = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00","99") else "01"
    dd = s[5:7] if s[5:7] not in ("00","99") else "01"
    try: return pd.Timestamp(f"{y}-{mm}-{dd}")
    except: return pd.NaT


def build_sequences(min_site_n=30):
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = df.columns.str.strip().str.lstrip('﻿')
    df["pid"]  = df["病歷號(2)"].astype(str).str.strip()
    df["site"] = df["腫瘤部位(47)"].astype(str).str[:3].str.upper()
    df["dx"]   = df["最初診斷日(45)"].apply(roc_to_ts)
    df["age"]  = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["sex"]  = df["性別(5)"].map({1:"M",2:"F","1":"M","2":"F"})
    df = df.dropna(subset=["dx","age","sex"])
    site_counts = df.groupby("site")["pid"].nunique()
    keep  = set(site_counts[site_counts >= min_site_n].index)
    df    = df[df["site"].isin(keep)]
    all_sites = sorted(keep)
    vocab = {s: i + SITE_OFFSET for i, s in enumerate(all_sites)}
    first = (df.sort_values("dx")
               .groupby(["pid","site"], as_index=False)
               .agg(dx=("dx","first"), age=("age","first"), sex=("sex","first")))
    first = first.sort_values(["pid","dx"])
    age_mean = first["age"].mean(); age_std = first["age"].std() + 1e-6
    seqs = []
    for pid, grp in first.groupby("pid"):
        grp  = grp.sort_values("dx")
        t0   = grp["dx"].iloc[0]
        seqs.append({
            "pid":      pid,
            "tokens":   [vocab[s] for s in grp["site"]],
            "days":     [(d - t0).days for d in grp["dx"]],
            "dx_dates": list(grp["dx"]),
            "sites":    list(grp["site"]),
            "ages":     [(a - age_mean) / age_std for a in grp["age"]],
            "sex_bin":  1.0 if grp["sex"].iloc[0] == "M" else 0.0,
            "n_cancers":len(grp),
        })
    return seqs, vocab


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
        enc[..., 1::2] = torch.cos(t * div[: self.d_model // 2])
        return enc


class CancerTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
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
            nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, vocab_size))
    def forward(self, tokens, days, ages, sex_bin, attn_mask):
        x = self.site_embed(tokens)
        x = x + self.time_enc(days)
        x = x + self.age_proj(ages.unsqueeze(-1))
        x = x + self.sex_embed(sex_bin.long()).unsqueeze(1)
        x = self.input_norm(x)
        h = self.encoder(x, src_key_padding_mask=~attn_mask)
        return self.mlm_head(h), h[:, 0, :]


def predict_next_from_first(model, first_token, first_age, sex_b, inv_vocab):
    """Query [CLS, A, MASK] → top-5 site probabilities for next cancer."""
    tokens = [CLS_IDX, first_token, MASK_IDX] + [PAD_IDX] * (MAX_SEQ - 3)
    days   = [0.0, 0.0, 0.0]           + [0.0] * (MAX_SEQ - 3)
    ages   = [first_age, first_age, 0.0] + [0.0] * (MAX_SEQ - 3)
    attn   = [True, True, True]         + [False] * (MAX_SEQ - 3)

    with torch.no_grad():
        t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        d = torch.tensor(days,   dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(ages,   dtype=torch.float32).unsqueeze(0)
        s = torch.tensor(float(sex_b), dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(attn,   dtype=torch.bool).unsqueeze(0)
        logits, _ = model(t, d, a, s, m)
        probs = torch.softmax(logits[0, 2, :], dim=-1).clone()

    # Zero out special tokens and the primary site itself
    for bad in [PAD_IDX, MASK_IDX, CLS_IDX, first_token]:
        probs[bad] = 0.0

    top5 = torch.argsort(probs, descending=True)[:5]
    return [(inv_vocab[tok.item()], float(probs[tok].item()))
            for tok in top5 if tok.item() in inv_vocab]


def compute_timing_windows(train_seqs, vocab):
    """Empirical gap distributions from training multi-cancer patients."""
    inv_vocab = {v: k for k, v in vocab.items()}
    records = []
    for s in train_seqs:
        if s["n_cancers"] < 2: continue
        sites = s["sites"]; days = s["days"]
        # All consecutive pairs
        for i in range(len(sites) - 1):
            records.append({
                "first_site":  sites[i],
                "second_site": sites[i + 1],
                "gap_days":    days[i + 1] - days[i],
            })
    df = pd.DataFrame(records)
    windows = (df.groupby(["first_site","second_site"])["gap_days"]
                  .agg(n_pairs="count",
                       gap_p25=lambda x: x.quantile(0.25),
                       gap_median="median",
                       gap_p75=lambda x: x.quantile(0.75))
                  .reset_index())
    windows.to_csv(OUT / "timing_windows.csv", index=False)
    return windows


def main():
    print("=== Registry DL — 12: Surveillance Calendar ===")

    # Build sequences
    print("  Building patient sequences…")
    seqs, vocab = build_sequences()
    inv_vocab   = {v: k for k, v in vocab.items()}
    vocab_size  = len(vocab) + SITE_OFFSET

    # Reconstruct same val split as Script 07
    multi_seqs = [s for s in seqs if s["n_cancers"] >= 2]
    rng        = np.random.default_rng(SEED)
    idx        = rng.permutation(len(multi_seqs))
    n_val      = int(len(multi_seqs) * VAL_FRAC)
    val_idx    = set(idx[:n_val].tolist())
    val_seqs   = [multi_seqs[i] for i in idx[:n_val]]
    train_seqs = [multi_seqs[i] for i in idx[n_val:]]
    print(f"  Train: {len(train_seqs):,}  Val: {len(val_seqs):,}")

    # Timing windows from training set
    print("  Computing timing windows…")
    timing_df = compute_timing_windows(train_seqs, vocab)
    overall_median = timing_df["gap_median"].median()
    overall_p25    = timing_df["gap_p25"].median()
    overall_p75    = timing_df["gap_p75"].median()

    # Load model
    print("  Loading transformer weights…")
    ckpt  = torch.load(MOUT / "transformer_weights.pt", map_location="cpu")
    model = CancerTransformer(vocab_size)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Generate calendar for each validation patient
    print("  Running inference on validation patients…")
    rows = []
    for s in val_seqs:
        first_token = s["tokens"][0]
        first_site  = s["sites"][0]
        first_age   = s["ages"][0]
        sex_b       = s["sex_bin"]

        # For 2-cancer patients: second cancer is the target
        true_second = s["sites"][1] if s["n_cancers"] >= 2 else None
        actual_gap  = s["days"][1]  if s["n_cancers"] >= 2 else None

        preds = predict_next_from_first(model, first_token, first_age, sex_b, inv_vocab)
        if not preds: continue

        pred_sites = [p[0] for p in preds]
        pred_probs = [p[1] for p in preds]

        # Pad to 5
        while len(pred_sites) < 5: pred_sites.append(""); pred_probs.append(0.0)

        hit1 = int(true_second == pred_sites[0]) if true_second else None
        hit3 = int(true_second in pred_sites[:3]) if true_second else None
        hit5 = int(true_second in pred_sites[:5]) if true_second else None

        # Timing window for top prediction
        pair = timing_df[
            (timing_df["first_site"] == first_site) &
            (timing_df["second_site"] == pred_sites[0])]
        if len(pair):
            t_p25 = float(pair["gap_p25"].iloc[0])
            t_med = float(pair["gap_median"].iloc[0])
            t_p75 = float(pair["gap_p75"].iloc[0])
        else:
            t_p25, t_med, t_p75 = overall_p25, overall_median, overall_p75

        rows.append({
            "pid":             s["pid"],
            "first_site":      first_site,
            "true_second_site":true_second,
            "actual_gap_days": actual_gap,
            "pred1": pred_sites[0], "prob1": round(pred_probs[0], 4),
            "pred2": pred_sites[1], "prob2": round(pred_probs[1], 4),
            "pred3": pred_sites[2], "prob3": round(pred_probs[2], 4),
            "pred4": pred_sites[3], "prob4": round(pred_probs[3], 4),
            "pred5": pred_sites[4], "prob5": round(pred_probs[4], 4),
            "hit@1": hit1, "hit@3": hit3, "hit@5": hit5,
            "timing_p25_days":    round(t_p25),
            "timing_median_days": round(t_med),
            "timing_p75_days":    round(t_p75),
        })

    cal_df = pd.DataFrame(rows)
    cal_df.to_csv(OUT / "surveillance_calendar.csv", index=False)

    # Summary
    has_true = cal_df["true_second_site"].notna()
    n_eval   = has_true.sum()
    r1 = cal_df.loc[has_true, "hit@1"].mean()
    r3 = cal_df.loc[has_true, "hit@3"].mean()
    r5 = cal_df.loc[has_true, "hit@5"].mean()
    print(f"\n  Surveillance calendar: {len(cal_df):,} patients")
    print(f"  Next-cancer prediction (first-only context):")
    print(f"    R@1={r1:.3f}  R@3={r3:.3f}  R@5={r5:.3f}")
    print(f"  Timing windows: {len(timing_df):,} site pairs")
    print(f"  Overall median gap: {overall_median:.0f} days  IQR: {overall_p25:.0f}–{overall_p75:.0f}")
    print(f"  Saved → {OUT}/")


if __name__ == "__main__":
    main()
