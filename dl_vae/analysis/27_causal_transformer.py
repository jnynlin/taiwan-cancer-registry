"""
Registry DL — Script 27: Causal (GPT-style) Cancer Sequence Transformer

Same architecture as Script 07 (BERT bidirectional) but with a lower-triangular
causal attention mask so position i can only attend to positions 0..i.

Training objective: next-cancer prediction
  Input:  [CLS, c₁, c₂, …, cₖ, MASK, PAD…]
  Predict: cₖ₊₁ at the MASK position
  With causal mask, MASK only sees [CLS, c₁…cₖ]; cannot peek ahead.

For each multi-cancer patient with n sites → n-1 training examples:
  k=1: [CLS, c₁, MASK]           → predict c₂
  k=2: [CLS, c₁, c₂, MASK]      → predict c₃
  …

Expected behaviour (vs Script 07 BERT):
  k=1: R@1 comparable to Script 12 (0.232) — same causal framing
  k=2: R@1 IMPROVES over k=1 (model can use temporal order)
  k=3: R@1 further improves up to data sparsity limit

Architecture: identical to Script 07
  D_MODEL=64, N_HEAD=4, N_LAYERS=2, FF_DIM=128, MAX_SEQ=8, norm_first=True

Outputs:
  models/causal_transformer_weights.pt
  results/27_causal/fig_train_loss.png
  results/27_causal/fig_recall_at_k.png
  results/27_causal/val_predictions.csv
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
MAX_SEQ     = 8
DROPOUT     = 0.1
BATCH_SIZE  = 256
LR          = 1e-3
MAX_EPOCHS  = 200
PATIENCE    = 20
VAL_FRAC    = 0.20
SEED        = 42

PAD_IDX   = 0
MASK_IDX  = 1
CLS_IDX   = 2
SITE_OFFSET = 3

BASE  = Path(__file__).parent.parent
RAW   = BASE.parent / "data/processed/all_cancers.csv"
DOUT  = BASE / "data"
MOUT  = BASE / "models"
OUT   = BASE / "results/27_causal"
OUT.mkdir(parents=True, exist_ok=True)
MOUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED)


# ── Data loading (identical to Script 07) ─────────────────────────────────────

def roc_to_ts(x):
    s = str(x).split(".")[0].zfill(7)
    if not s.isdigit() or len(s) != 7: return pd.NaT
    y = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00","99") else "01"
    dd = s[5:7] if s[5:7] not in ("00","99") else "01"
    try: return pd.Timestamp(f"{y}-{mm}-{dd}")
    except: return pd.NaT


def build_sequences(min_site_n=30):
    print("  Loading registry…")
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
    vocab     = {s: i + SITE_OFFSET for i, s in enumerate(all_sites)}

    # (pid, site) dedup — first occurrence only
    first = (df.sort_values("dx")
               .groupby(["pid","site"], as_index=False)
               .agg(dx=("dx","first"), age=("age","first"), sex=("sex","first")))
    first = first.sort_values(["pid","dx"])

    age_mean = first["age"].mean()
    age_std  = first["age"].std() + 1e-6

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

    print(f"  {len(seqs):,} patients · {len(vocab)} sites")
    print(f"  Multi-cancer (≥2): {sum(1 for s in seqs if s['n_cancers']>=2):,}")
    return seqs, vocab, age_mean, age_std


# ── Dataset (causal: next-cancer prediction) ──────────────────────────────────

class CausalSeqDataset(Dataset):
    """
    For each multi-cancer patient with n cancers → n-1 examples.
    Example k: input=[CLS, c₁…cₖ, MASK], label=c_{k+1}
    MASK position = k+1 (after CLS and k context cancers).
    """
    def __init__(self, seqs):
        self.examples = []
        for s in seqs:
            n = min(len(s["tokens"]), MAX_SEQ - 2)  # -2: CLS + MASK
            if n < 1:
                continue
            for k in range(1, n + 1):               # k = context length
                if k >= len(s["tokens"]):
                    break
                self.examples.append((s, k))

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        s, k = self.examples[idx]
        # [CLS, c₁, …, cₖ, MASK, PAD…]
        ctx_tokens = s["tokens"][:k]
        ctx_days   = s["days"][:k]
        ctx_ages   = s["ages"][:k]
        target     = s["tokens"][k]        # c_{k+1} — the next cancer

        tokens = [CLS_IDX] + ctx_tokens + [MASK_IDX]
        days   = [0.0]     + ctx_days   + [float(ctx_days[-1]) + 1 if ctx_days else 1.0]
        ages   = [s["ages"][0]] + ctx_ages + [0.0]

        seq_len = len(tokens)
        pad_len = MAX_SEQ - seq_len
        tokens += [PAD_IDX] * pad_len
        days   += [0.0]     * pad_len
        ages   += [0.0]     * pad_len

        labels      = [-100] * MAX_SEQ
        mask_pos    = k + 1             # position of MASK in padded sequence
        labels[mask_pos] = target

        attn = [True] * seq_len + [False] * pad_len

        return (torch.tensor(tokens,  dtype=torch.long),
                torch.tensor(days,    dtype=torch.float32),
                torch.tensor(ages,    dtype=torch.float32),
                torch.tensor(s["sex_bin"], dtype=torch.float32),
                torch.tensor(attn,    dtype=torch.bool),
                torch.tensor(labels,  dtype=torch.long))


# ── Model ─────────────────────────────────────────────────────────────────────

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
    def __init__(self, vocab_size, d_model, nhead, num_layers, ff_dim, dropout,
                 causal=False):
        super().__init__()
        self.causal      = causal
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
        x  = self.site_embed(tokens)
        x  = x + self.time_enc(days)
        x  = x + self.age_proj(ages.unsqueeze(-1))
        x  = x + self.sex_embed(sex_bin.long()).unsqueeze(1)
        x  = self.input_norm(x)
        pad_mask = ~attn_mask

        if self.causal:
            S = tokens.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                S, device=tokens.device)
        else:
            causal_mask = None

        h = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        return self.mlm_head(h), h[:, 0, :]


def mlm_loss(logits, labels):
    B, S, V = logits.shape
    mask = (labels != -100).view(-1)
    if mask.sum() == 0:
        return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()
    return nn.functional.cross_entropy(
        logits.view(B * S, V)[mask], labels.view(-1)[mask])


# ── Evaluation helpers (same as Script 07/08) ─────────────────────────────────

def recall_at_k(model, val_seqs, vocab, device, ks=(1,3,5)):
    """
    Causal inference: predict next cancer from context of length k.
    Evaluates at context lengths 1..MAX_SEQ-2.
    Returns dict of R@k for context length k=1 (primary metric).
    """
    inv_vocab = {v: k for k, v in vocab.items()}
    model.eval()
    hits = {k: 0 for k in ks}
    total = 0
    with torch.no_grad():
        for s in val_seqs:
            n = min(s["n_cancers"], MAX_SEQ - 1)
            if n < 2:
                continue
            # Use k=1 (first cancer only) as primary eval — comparable to Script 12
            k = 1
            ctx_tokens = s["tokens"][:k]
            ctx_days   = s["days"][:k]
            ctx_ages   = s["ages"][:k]
            target_tok = s["tokens"][k]
            target_site = inv_vocab.get(target_tok)
            if target_site is None:
                continue

            # Build input
            toks  = [CLS_IDX] + ctx_tokens + [MASK_IDX]
            days  = [0.0] + ctx_days + [float(ctx_days[-1]) + 1]
            ages  = [s["ages"][0]] + ctx_ages + [0.0]
            seq_len = len(toks)
            pad_len = MAX_SEQ - seq_len
            toks += [PAD_IDX] * pad_len
            days += [0.0]     * pad_len
            ages += [0.0]     * pad_len
            attn = [True] * seq_len + [False] * pad_len
            mask_pos = k + 1

            t = torch.tensor([toks],  dtype=torch.long,    device=device)
            d = torch.tensor([days],  dtype=torch.float32, device=device)
            a = torch.tensor([ages],  dtype=torch.float32, device=device)
            sx = torch.tensor([s["sex_bin"]], dtype=torch.float32, device=device)
            m = torch.tensor([attn],  dtype=torch.bool,    device=device)

            logits, _ = model(t, d, a, sx, m)
            probs = torch.softmax(logits[0, mask_pos, :], dim=-1).clone()
            for bad in [PAD_IDX, MASK_IDX, CLS_IDX] + ctx_tokens:
                if bad < probs.size(0):
                    probs[bad] = 0.0

            top_k = torch.argsort(probs, descending=True)
            total += 1
            for rank_k in ks:
                preds = [inv_vocab.get(top_k[i].item()) for i in range(rank_k)]
                if target_site in preds:
                    hits[rank_k] += 1

    return {k: hits[k] / total if total else 0 for k in ks}, total


# ── Training ───────────────────────────────────────────────────────────────────

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Building sequences…")
    seqs, vocab, age_mean, age_std = build_sequences()
    vocab_size = len(vocab) + SITE_OFFSET

    # Train/val split — identical to Script 07 (same SEED, same logic)
    multi = [s for s in seqs if s["n_cancers"] >= 2]
    rng   = np.random.default_rng(SEED)
    idx   = rng.permutation(len(multi))
    n_val = int(len(multi) * VAL_FRAC)
    val_seqs   = [multi[i] for i in idx[:n_val]]
    train_seqs = [multi[i] for i in idx[n_val:]]
    print(f"  Train multi: {len(train_seqs):,}  Val multi: {len(val_seqs):,}")

    train_ds = CausalSeqDataset(train_seqs)
    val_ds   = CausalSeqDataset(val_seqs)
    print(f"  Train examples: {len(train_ds):,}  Val examples: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = CancerTransformer(
        vocab_size=vocab_size, d_model=D_MODEL, nhead=N_HEAD,
        num_layers=N_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT,
        causal=True).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=MAX_EPOCHS, eta_min=LR * 0.01)

    best_val   = float("inf")
    patience_c = 0
    train_hist, val_hist = [], []

    print(f"\nTraining (causal)  max_epochs={MAX_EPOCHS}  patience={PATIENCE}")
    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──
        model.train()
        t_loss = 0.0
        for batch in train_loader:
            toks, days, ages, sex, attn, labels = [b.to(device) for b in batch]
            optimiser.zero_grad()
            logits, _ = model(toks, days, ages, sex, attn)
            loss = mlm_loss(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            t_loss += loss.item()
        scheduler.step()
        t_loss /= len(train_loader)

        # ── val ──
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                toks, days, ages, sex, attn, labels = [b.to(device) for b in batch]
                logits, _ = model(toks, days, ages, sex, attn)
                v_loss += mlm_loss(logits, labels).item()
        v_loss /= len(val_loader)

        train_hist.append(t_loss)
        val_hist.append(v_loss)

        if epoch % 10 == 0 or epoch == 1:
            recalls, n_eval = recall_at_k(model, val_seqs, vocab, device)
            r1, r3, r5 = recalls[1], recalls[3], recalls[5]
            print(f"  ep{epoch:3d}  t={t_loss:.4f}  v={v_loss:.4f}  "
                  f"R@1={r1:.3f} R@3={r3:.3f} R@5={r5:.3f}  (n={n_eval})")

        if v_loss < best_val:
            best_val   = v_loss
            patience_c = 0
            torch.save({"model":       model.state_dict(),
                        "vocab":       vocab,
                        "vocab_size":  vocab_size,
                        "age_mean":    float(age_mean),
                        "age_std":     float(age_std),
                        "causal":      True,
                        "d_model":     D_MODEL,
                        "n_head":      N_HEAD,
                        "n_layers":    N_LAYERS,
                        "ff_dim":      FF_DIM,
                        "max_seq":     MAX_SEQ},
                       MOUT / "causal_transformer_weights.pt")
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (best val={best_val:.4f})")
                break

    # ── Final eval with best checkpoint ───────────────────────────────────────
    ckpt  = torch.load(MOUT / "causal_transformer_weights.pt",
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    recalls, n_eval = recall_at_k(model, val_seqs, vocab, device, ks=(1,3,5))
    r1, r3, r5 = recalls[1], recalls[3], recalls[5]
    print(f"\nFinal (best ckpt): R@1={r1:.3f}  R@3={r3:.3f}  R@5={r5:.3f}  n={n_eval}")
    print(f"Random baseline:   R@1={1/len(vocab):.4f}")

    # Save val predictions
    rows = []
    model.eval()
    inv_vocab = {v: k for k, v in vocab.items()}
    with torch.no_grad():
        for s in val_seqs:
            n = min(s["n_cancers"], MAX_SEQ - 1)
            if n < 2:
                continue
            k = 1
            ctx_tokens = s["tokens"][:k]
            ctx_days   = s["days"][:k]
            ctx_ages   = s["ages"][:k]
            target_tok = s["tokens"][k]
            target     = inv_vocab.get(target_tok)
            if target is None:
                continue

            toks = [CLS_IDX] + ctx_tokens + [MASK_IDX]
            days_inp = [0.0] + ctx_days + [float(ctx_days[-1]) + 1]
            ages_inp = [s["ages"][0]] + ctx_ages + [0.0]
            seq_len  = len(toks)
            pad_len  = MAX_SEQ - seq_len
            toks    += [PAD_IDX] * pad_len
            days_inp += [0.0] * pad_len
            ages_inp += [0.0] * pad_len
            attn = [True]*seq_len + [False]*pad_len
            mask_pos = k + 1

            t  = torch.tensor([toks],    dtype=torch.long,    device=device)
            d  = torch.tensor([days_inp],dtype=torch.float32, device=device)
            a  = torch.tensor([ages_inp],dtype=torch.float32, device=device)
            sx = torch.tensor([s["sex_bin"]],dtype=torch.float32, device=device)
            m  = torch.tensor([attn],    dtype=torch.bool,    device=device)

            logits, _ = model(t, d, a, sx, m)
            probs = torch.softmax(logits[0, mask_pos, :], dim=-1).clone()
            for bad in [PAD_IDX, MASK_IDX, CLS_IDX] + ctx_tokens:
                if bad < probs.size(0):
                    probs[bad] = 0.0
            top5 = torch.argsort(probs, descending=True)[:5]
            preds = [inv_vocab.get(tok.item(), "?") for tok in top5]
            probs5 = [probs[tok].item() for tok in top5]
            rows.append({
                "pid": s["pid"],
                "first_site": inv_vocab.get(ctx_tokens[0], "?"),
                "true_second": target,
                "pred1": preds[0], "prob1": round(probs5[0], 4),
                "pred2": preds[1], "prob2": round(probs5[1], 4),
                "pred3": preds[2], "prob3": round(probs5[2], 4),
                "hit1": target == preds[0],
                "hit3": target in preds[:3],
                "hit5": target in preds[:5],
            })

    pd.DataFrame(rows).to_csv(OUT / "val_predictions.csv", index=False)

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(train_hist, label="Train", color="#2563eb")
    ax.plot(val_hist,   label="Val",   color="#dc2626")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Causal Transformer — Training Loss")
    ax.legend()

    # R@k by context length from final model
    ax = axes[1]
    r1s, ns = [], []
    for k_ctx in range(1, MAX_SEQ - 1):
        hits, total = 0, 0
        for s in val_seqs:
            if s["n_cancers"] <= k_ctx:
                continue
            ctx_tokens = s["tokens"][:k_ctx]
            ctx_days   = s["days"][:k_ctx]
            target_tok = s["tokens"][k_ctx]
            target     = inv_vocab.get(target_tok)
            if target is None:
                continue
            toks = [CLS_IDX] + ctx_tokens + [MASK_IDX]
            d2   = [0.0] + ctx_days + [float(ctx_days[-1]) + 1 if ctx_days else 1.0]
            a2   = [s["ages"][0]] + s["ages"][:k_ctx] + [0.0]
            sl   = len(toks)
            pl   = MAX_SEQ - sl
            toks += [PAD_IDX]*pl; d2 += [0.0]*pl; a2 += [0.0]*pl
            atn  = [True]*sl + [False]*pl
            mp   = k_ctx + 1
            with torch.no_grad():
                t2  = torch.tensor([toks], dtype=torch.long,    device=device)
                dd  = torch.tensor([d2],   dtype=torch.float32, device=device)
                aa  = torch.tensor([a2],   dtype=torch.float32, device=device)
                sx2 = torch.tensor([s["sex_bin"]], dtype=torch.float32, device=device)
                mm  = torch.tensor([atn],  dtype=torch.bool,    device=device)
                lg, _ = model(t2, dd, aa, sx2, mm)
                probs2 = torch.softmax(lg[0, mp, :], dim=-1).clone()
                for bad in [PAD_IDX, MASK_IDX, CLS_IDX] + ctx_tokens:
                    if bad < probs2.size(0): probs2[bad] = 0.0
                top1 = inv_vocab.get(torch.argmax(probs2).item())
            total += 1
            if top1 == target:
                hits += 1
        if total < 5:
            break
        r1s.append(hits / total)
        ns.append(total)

    ax.bar(range(1, len(r1s)+1), r1s, color="#2563eb", alpha=0.8)
    ax.axhline(1/len(vocab), color="#9ca3af", linestyle="--", label="Random baseline")
    ax.axhline(0.232, color="#dc2626", linestyle=":", label="Script 12 (BERT k=1, 0.232)")
    ax.set(xlabel="Context length k", ylabel="R@1",
           title="Causal Transformer — R@1 by Context Length",
           xticks=range(1, len(r1s)+1),
           xticklabels=[f"k={i}\nn={ns[i-1]}" for i in range(1, len(r1s)+1)])
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_causal_training.png", dpi=150)
    plt.close(fig)

    print(f"\nOutputs → {OUT}/")
    print(f"Checkpoint → {MOUT}/causal_transformer_weights.pt")
    return r1, n_eval


if __name__ == "__main__":
    train()
