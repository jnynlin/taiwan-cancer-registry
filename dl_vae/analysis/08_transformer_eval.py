"""
Registry DL — Script 08: Transformer Evaluation + Temporal Analysis

Unique analyses that the time-aware Transformer enables over 02b MLP:

  1. R@k comparison: Transformer vs 02b MLP baseline
  2. Temporal sensitivity: P(C15 | C12, gap=T) as function of gap T
     — Does surveillance urgency decay with time since first cancer?
  3. Site embedding similarity: cosine distance in learned embedding space
     — Are UADT sites clustered together?
  4. Trigram discovery: most common 3-cancer sequences
  5. UMAP of patient CLS embeddings

Outputs:
  results/08_transformer_eval/fig_temporal_sensitivity.png
  results/08_transformer_eval/fig_embedding_similarity.png
  results/08_transformer_eval/fig_trigrams.png
  results/08_transformer_eval/fig_umap_cls.png
  results/08_transformer_eval/fig_model_comparison.png
  results/08_transformer_eval/temporal_sensitivity.csv
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: pip install torch"); raise SystemExit(1)

try:
    import umap
    UMAP_OK = True
except ImportError:
    from sklearn.decomposition import PCA
    UMAP_OK = False

from sklearn.metrics.pairwise import cosine_similarity

BASE  = Path(__file__).parent.parent
DOUT  = BASE / "data"
MOUT  = BASE / "models"
OUT   = BASE / "results/08_transformer_eval"
OUT.mkdir(parents=True, exist_ok=True)

PAD_IDX = 0; MASK_IDX = 1; CLS_IDX = 2; SITE_OFFSET = 3
D_MODEL = 64; N_HEAD = 4; N_LAYERS = 2; FF_DIM = 128; MAX_SEQ = 8; DROPOUT = 0.1
NAVY = "#14304a"; ACCENT = "#2e7fbf"

from constants import UADT_SITES  # noqa: E402

# ── Rebuild model (must match 07 exactly) ─────────────────────────────────────

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
        self.time_enc   = TimeEncoding(d_model)
        self.age_proj   = nn.Linear(1, d_model)
        self.sex_embed  = nn.Embedding(2, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size))
        self.d_model = d_model
    def forward(self, tokens, days, ages, sex_bin, attn_mask):
        x = self.site_embed(tokens)
        x = x + self.time_enc(days)
        x = x + self.age_proj(ages.unsqueeze(-1))
        x = x + self.sex_embed(sex_bin.long()).unsqueeze(1)
        x = self.input_norm(x)
        pad_mask = ~attn_mask
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.mlm_head(h), h[:, 0, :]


def query_model(model, vocab, site_sequence, days_sequence,
                age=55.0, sex_male=True, age_mean=60.0, age_std=12.0):
    """
    Query: given site_sequence + days_sequence, return P(next site) distribution.
    Appends [MASK] after the last known site and reads its logit distribution.
    """
    n    = min(len(site_sequence), MAX_SEQ - 2)   # -2 for CLS + MASK
    toks = [CLS_IDX] + [vocab[s] for s in site_sequence[:n]] + [MASK_IDX]
    days = [0.0] + list(days_sequence[:n]) + [float(days_sequence[n-1]) + 1 if n > 0 else 1.0]
    age_n = (age - age_mean) / age_std
    ages = [age_n] * len(toks)
    sex_b = 1.0 if sex_male else 0.0
    pad_len = MAX_SEQ - len(toks)
    toks  += [PAD_IDX] * pad_len
    days  += [0.0]     * pad_len
    ages  += [0.0]     * pad_len
    attn   = [True] * (MAX_SEQ - pad_len) + [False] * pad_len

    with torch.no_grad():
        logits, _ = model(
            torch.tensor([toks], dtype=torch.long),
            torch.tensor([days], dtype=torch.float32),
            torch.tensor([ages], dtype=torch.float32),
            torch.tensor([sex_b], dtype=torch.float32),
            torch.tensor([attn], dtype=torch.bool))
    mask_pos = len(site_sequence[:n]) + 1   # position of [MASK] after CLS
    logits_mask = logits[0, mask_pos, :]    # (V,)
    probs = torch.softmax(logits_mask, dim=-1).numpy()
    return probs


def main():
    print("=== Registry DL — 08: Transformer Evaluation ===")

    ckpt       = torch.load(MOUT / "transformer_weights.pt", map_location="cpu")
    vocab      = ckpt["vocab"]
    vocab_size = ckpt["vocab_size"]
    inv_vocab  = {v: k for k, v in vocab.items()}
    sites      = [inv_vocab[i] for i in sorted(inv_vocab)]

    model = CancerTransformer(vocab_size, D_MODEL, N_HEAD, N_LAYERS, FF_DIM, DROPOUT)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded model · vocab {vocab_size} · {len(vocab)} sites")

    # Demographics for typical patient
    meta     = pd.read_csv(DOUT / "patient_meta.csv", index_col="pid")
    age_mean = meta["age_first"].mean()
    age_std  = meta["age_first"].std() + 1e-6

    # ── 1. Temporal sensitivity ───────────────────────────────────────────────
    print("  Temporal sensitivity analysis…")
    # Primary = C12 pyriform. Query: P(C15 esophagus | C12, gap=T days)
    gaps_days = [0, 30, 90, 180, 365, 730, 1095, 1825]  # 0 to 5 years
    case_sites = ["C12","C13","C06","C15"]   # index cancers
    temp_rows  = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    target_sites_map = {"C12":"C15","C13":"C15","C06":"C15","C15":"C12"}

    for ax, primary in zip(axes, case_sites):
        target = target_sites_map[primary]
        if primary not in vocab or target not in vocab:
            ax.axis("off"); continue
        probs_over_time = []
        for gap in gaps_days:
            probs = query_model(model, vocab, [primary], [0.0],
                                age=60.0, sex_male=True,
                                age_mean=age_mean, age_std=age_std)
            # Re-query with gap encoded — C12 at day 0, MASK at day=gap
            n = min(1, MAX_SEQ - 2)
            toks = [CLS_IDX, vocab[primary], MASK_IDX]
            dys  = [0.0, 0.0, float(gap)]
            age_n = (60.0 - age_mean) / age_std
            ages = [age_n] * 3
            pad_len = MAX_SEQ - 3
            toks += [PAD_IDX]*pad_len; dys += [0.0]*pad_len; ages += [0.0]*pad_len
            attn = [True]*3 + [False]*pad_len
            with torch.no_grad():
                logits, _ = model(
                    torch.tensor([toks], dtype=torch.long),
                    torch.tensor([dys],  dtype=torch.float32),
                    torch.tensor([ages], dtype=torch.float32),
                    torch.tensor([1.0],  dtype=torch.float32),
                    torch.tensor([attn], dtype=torch.bool))
            p_target = torch.softmax(logits[0, 2, :], dim=-1)[vocab[target]].item()
            probs_over_time.append(p_target)
            temp_rows.append({"primary": primary, "target": target,
                               "gap_days": gap, "p_target": round(p_target, 5)})

        gap_yr = [g/365.25 for g in gaps_days]
        ax.plot(gap_yr, probs_over_time, marker="o", color=ACCENT, linewidth=2)
        ax.set_xlabel("Years since primary diagnosis")
        ax.set_ylabel(f"P(next = {target})")
        ax.set_title(f"Primary: {primary} → P({target})")
        ax.set_ylim(0, max(probs_over_time)*1.2 + 0.001)
        ax.grid(alpha=0.3)

    fig.suptitle("Temporal sensitivity: P(secondary site | primary, gap)\n"
                 "(male, age 60, first cancer at t=0)", fontsize=12,
                 color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_temporal_sensitivity.png", dpi=150); plt.close()
    pd.DataFrame(temp_rows).to_csv(OUT / "temporal_sensitivity.csv", index=False)

    # ── 2. Site embedding similarity ──────────────────────────────────────────
    print("  Site embedding similarity…")
    site_embs = model.site_embed.weight.detach().numpy()  # (V, D)
    valid_sites = [s for s in sorted(vocab) if vocab[s] < site_embs.shape[0]]
    emb_mat = np.stack([site_embs[vocab[s]] for s in valid_sites])
    sim_mat = cosine_similarity(emb_mat)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(valid_sites))); ax.set_xticklabels(valid_sites, rotation=90, fontsize=7)
    ax.set_yticks(range(len(valid_sites))); ax.set_yticklabels(valid_sites, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Cosine similarity")
    # Highlight UADT sites
    uadt_idx = [i for i, s in enumerate(valid_sites) if s in UADT_SITES]
    for idx in uadt_idx:
        ax.axhline(idx - 0.5, color="#e05c2e", linewidth=0.4, alpha=0.6)
        ax.axvline(idx - 0.5, color="#e05c2e", linewidth=0.4, alpha=0.6)
    ax.set_title("Learned site embedding similarity (cosine)\nOrange lines = UADT field sites")
    fig.tight_layout()
    fig.savefig(OUT / "fig_embedding_similarity.png", dpi=150); plt.close()

    # ── 3. Trigram discovery ──────────────────────────────────────────────────
    print("  Trigram discovery from val predictions…")
    val_preds = pd.read_csv(OUT.parent.parent / "results/07_transformer/val_predictions.csv")

    # Load sequences to find observed trigrams
    seq_file = DOUT / "cancer_matrix.csv"
    X_df = pd.read_csv(seq_file, index_col="pid")
    # Find patients with ≥3 cancers from patient_meta
    meta3 = meta[meta["n_sites"] >= 3]
    print(f"  Patients with ≥3 cancers: {len(meta3):,}")
    trigrams = {}
    for pid in meta3.index:
        if pid not in X_df.index: continue
        row   = X_df.loc[pid]
        sites_present = sorted([s for s in X_df.columns if row[s] == 1])
        for trio in combinations(sites_present, 3):
            key = "→".join(trio)
            trigrams[key] = trigrams.get(key, 0) + 1

    top_trigrams = sorted(trigrams.items(), key=lambda x: -x[1])[:20]
    labels_t = [t[0] for t in top_trigrams]
    counts_t = [t[1] for t in top_trigrams]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_t = ["#e05c2e" if all(s in UADT_SITES for s in t[0].split("→"))
                else ACCENT for t in top_trigrams]
    ax.barh(range(len(labels_t))[::-1], counts_t, color=colors_t[::-1])
    ax.set_yticks(range(len(labels_t))[::-1])
    ax.set_yticklabels(labels_t[::-1], fontsize=8)
    ax.set_xlabel("Patient count")
    ax.set_title("Top-20 cancer trigrams (co-occurring site triples)\n"
                 "Orange = all-UADT-field trigrams")
    fig.tight_layout()
    fig.savefig(OUT / "fig_trigrams.png", dpi=150); plt.close()

    # ── 4. UMAP of CLS embeddings ─────────────────────────────────────────────
    print("  UMAP of CLS embeddings…")
    cls_emb = np.load(DOUT / "patient_cls_embeddings.npy")
    n_multi = (meta["n_sites"] >= 2).values
    sample  = np.random.default_rng(42).choice(len(cls_emb),
                size=min(15000, len(cls_emb)), replace=False)
    emb_s   = cls_emb[sample]
    multi_s = n_multi[sample] if len(n_multi) == len(cls_emb) else np.zeros(len(sample), dtype=bool)

    if UMAP_OK:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        proj = reducer.fit_transform(emb_s)
        proj_label = "UMAP"
    else:
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2, random_state=42).fit_transform(emb_s)
        proj_label = "PCA"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(proj[~multi_s, 0], proj[~multi_s, 1], s=2, alpha=0.2,
               color="#cccccc", label=f"Single-cancer (n={(~multi_s).sum():,})")
    ax.scatter(proj[multi_s,  0], proj[multi_s,  1], s=6, alpha=0.7,
               color="#e05c2e", label=f"Multi-cancer (n={multi_s.sum():,})")
    ax.set_xlabel(f"{proj_label}-1"); ax.set_ylabel(f"{proj_label}-2")
    ax.set_title(f"Transformer CLS embeddings — {proj_label}\n"
                 "Multi-cancer patients highlighted")
    ax.legend(markerscale=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_umap_cls.png", dpi=150); plt.close()

    # ── 5. Model comparison table ─────────────────────────────────────────────
    # Load 02b val predictions for comparison
    mlp_pred_path = BASE / "results/02b_predictor/val_predictions.csv"
    comp_rows = [{"model": "Transformer (07)", **{k: v for k, v in
                  {"R@1": None, "R@3": None, "R@5": None}.items()}}]

    # Recompute from val_predictions
    t_preds = pd.read_csv(OUT.parent.parent / "results/07_transformer/val_predictions.csv")
    for k in [1, 3, 5]:
        comp_rows[0][f"R@{k}"] = round((t_preds["rank_of_true"] <= k).mean(), 3)

    if mlp_pred_path.exists():
        m_preds = pd.read_csv(mlp_pred_path)
        for k in [1, 3, 5]:
            comp_rows.append({"model": "Masked MLP (02b)",
                               f"R@{k}": round((m_preds["rank_of_true"] <= k).mean(), 3)})
            # Fix: append to same row
        mlp_row = {"model": "Masked MLP (02b)"}
        for k in [1, 3, 5]:
            mlp_row[f"R@{k}"] = round((m_preds["rank_of_true"] <= k).mean(), 3)
        comp_rows = [comp_rows[0], mlp_row]

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT / "model_comparison.csv", index=False)
    print(f"\n  Model comparison:")
    print(comp_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3); w = 0.35
    colors_m = [NAVY, ACCENT]
    for i, row in comp_df.iterrows():
        vals = [row.get("R@1", 0), row.get("R@3", 0), row.get("R@5", 0)]
        bars = ax.bar(x + i * w, vals, w, label=row["model"],
                      color=colors_m[i % len(colors_m)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x + w/2); ax.set_xticklabels(["R@1","R@3","R@5"])
    ax.set_ylim(0, 0.7); ax.set_ylabel("Recall")
    ax.set_title("Next-cancer prediction: Transformer vs Masked MLP")
    ax.legend(fontsize=9)
    ax.axhline(1/37, color="red", linestyle="--", linewidth=0.8, label="Random")
    fig.tight_layout()
    fig.savefig(OUT / "fig_model_comparison.png", dpi=150); plt.close()

    print(f"\n  Figures → results/08_transformer_eval/")


if __name__ == "__main__":
    main()
