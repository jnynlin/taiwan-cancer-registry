"""
UADT Field Cancerization — Script 07: DeepHit Competing Risks

Replaces the Cox proportional hazards model (scripts 05/05b) with DeepHit,
a discrete-time neural network that:
  1. Makes NO proportional hazards assumption
  2. Models TWO competing events simultaneously:
       Cause 1: death without prior second UADT field cancer
       Cause 2: second UADT field cancer (new primary)
  3. Captures non-linear covariate effects (age × site × sex)

Dataset construction:
  - Cohort: all 10,113 UADT field patients (first cancer defines t=0)
  - Cause 2 takes precedence over cause 1 when t_second < t_death
  - Covariates: age_first, sex, first_site (one-hot, 10 sites)

Model: shared MLP trunk → 2 cause-specific heads → softmax over
       (N_TIMES × N_CAUSES) → joint P(T=t, K=k)

Loss: α·NLL + (1-α)·ranking_loss (DeepHit original formulation)

Outputs:
  results/07_deephit/competing_risks_data.csv  (no PIDs — aggregated only)
  results/07_deephit/fig_cif_curves.png        — CIF by age group + sex
  results/07_deephit/fig_risk_groups.png       — top/bottom quartile trajectories
  results/07_deephit/fig_covariate_effect.png  — age modulation of 3-yr CIF
  results/07_deephit/deephit_summary.csv       — C-td vs Cox comparison
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

# ── Constants ─────────────────────────────────────────────────────────────────
FIELD_SITES = ['C02','C03','C04','C05','C06','C09','C10','C12','C13','C15']
N_TIMES     = 30       # discrete time bins (percentile-based)
N_CAUSES    = 2        # cause 1=death, cause 2=second UADT cancer
ALPHA       = 0.2      # weight of ranking loss (vs NLL)
SIGMA       = 0.1      # ranking loss temperature
HIDDEN      = 64
CAUSE_DIM   = 32
BATCH_SIZE  = 256
LR          = 5e-4
MAX_EPOCHS  = 300
PATIENCE    = 30
VAL_FRAC    = 0.20
SEED        = 42
EVAL_HORIZONS_YR = [1, 3, 5]   # years for CIF evaluation

BASE  = Path(__file__).parent.parent
DATA  = BASE / "data"
OUT   = BASE / "results/07_deephit"
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED)

# ── Data preparation ──────────────────────────────────────────────────────────

def build_competing_risks():
    """
    Returns DataFrame with columns:
      t_event, cause (0=censor, 1=death, 2=second UADT cancer),
      age_norm, sex_bin, site_* (one-hot), age_first, sex, first_site
    """
    fp = pd.read_csv(DATA / "field_patients.csv")
    fm = pd.read_csv(DATA / "field_meta.csv")
    fp["dx"]     = pd.to_datetime(fp["dx"])
    fp["end_fu"] = pd.to_datetime(fp["end_fu"])

    # Per-patient: first dx, second dx (multi-field), end_fu, dead
    first_dx = fp.groupby("pid")["dx"].min().rename("first_dx")
    # First site = site at earliest dx
    first_site = (fp.sort_values("dx").groupby("pid")
                    .first()[["site","age","sex","dead","end_fu"]]
                    .rename(columns={"site":"first_site",
                                     "age":"age_first",
                                     "dead":"dead_first",
                                     "end_fu":"end_fu_first"}))
    # Second dx: second smallest dx per pid
    def second_dx(g):
        s = g.sort_values("dx")["dx"]
        return s.iloc[1] if len(s) > 1 else pd.NaT
    sec = fp.groupby("pid").apply(second_dx).rename("second_dx")

    df = first_site.join(first_dx).join(sec)
    df = df.join(fm.set_index("pid")[["multi_field","dead"]]
                    .rename(columns={"dead":"dead_meta"}))

    df["t_total"]  = (df["end_fu_first"] - df["first_dx"]).dt.days.clip(lower=1)
    df["t_second"] = (df["second_dx"]    - df["first_dx"]).dt.days.clip(lower=1)
    df["dead"]     = df["dead_meta"].fillna(df["dead_first"]).astype(int)

    # Competing event assignment
    def assign_event(row):
        if row["multi_field"] == 1 and pd.notna(row["t_second"]):
            t2 = row["t_second"]
            if row["dead"] == 1 and row["t_total"] <= t2:
                return row["t_total"], 1   # died before second cancer
            else:
                return t2, 2               # second cancer (cause 2)
        else:
            return row["t_total"], (1 if row["dead"] == 1 else 0)

    events = df.apply(assign_event, axis=1, result_type="expand")
    events.columns = ["t_event", "cause"]
    df = df.join(events)
    df = df[df["t_event"] > 0].copy()

    # Covariates
    age_mean = df["age_first"].mean(); age_std = df["age_first"].std() + 1e-6
    df["age_norm"] = (df["age_first"] - age_mean) / age_std
    df["sex_bin"]  = (df["sex"] == "M").astype(float)
    for s in FIELD_SITES:
        df[f"site_{s}"] = (df["first_site"] == s).astype(float)

    print(f"  Cohort: {len(df):,} patients")
    print(f"  Cause 0 (censored): {(df['cause']==0).sum():,}")
    print(f"  Cause 1 (death):    {(df['cause']==1).sum():,}")
    print(f"  Cause 2 (2nd UADT): {(df['cause']==2).sum():,}")

    # Discretize time using event-time quantiles
    event_times = df.loc[df["cause"] > 0, "t_event"].values
    cuts = np.unique(np.quantile(event_times, np.linspace(0, 1, N_TIMES + 1)))
    cuts[0] = 0; cuts[-1] = cuts[-1] + 1
    df["t_bin"] = np.digitize(df["t_event"], cuts[1:])   # 0-indexed bin

    return df, cuts, age_mean, age_std


# ── Model ─────────────────────────────────────────────────────────────────────

class DeepHit(nn.Module):
    def __init__(self, in_dim, n_times, n_causes, hidden, cause_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, cause_dim), nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cause_dim, cause_dim), nn.ReLU(),
                nn.Linear(cause_dim, n_times),
            ) for _ in range(n_causes)
        ])
        self.n_times  = n_times
        self.n_causes = n_causes

    def forward(self, x):
        h = self.shared(x)
        logits = torch.stack([head(h) for head in self.heads], dim=2)  # (B, T, K)
        # Joint softmax over all T×K outcomes
        flat   = logits.view(logits.size(0), -1)
        flat   = torch.log_softmax(flat, dim=1)
        return flat.view(logits.size(0), self.n_times, self.n_causes)   # log P(T=t, K=k)


def cif(log_F, cause_idx):
    """Cause-specific CIF: cumsum of P(T=t, K=cause) over t."""
    F = log_F[:, :, cause_idx].exp()          # (B, T)
    return F.cumsum(dim=1)                     # (B, T)


def overall_survival(log_F):
    """S(t) = 1 - sum_k CIF_k(t)."""
    return 1 - log_F.exp().sum(dim=2).cumsum(dim=1)   # (B, T)


# ── Losses ────────────────────────────────────────────────────────────────────

def nll_loss(log_F, t_bin, cause):
    """
    NLL:
      observed (cause>0): -log P(T=t_bin, K=cause-1)
      censored (cause=0): -log S(t_bin) = -log(1 - sum CIF up to t_bin)
    """
    B = log_F.size(0)
    obs_mask  = cause > 0
    cens_mask = ~obs_mask
    loss = torch.zeros(B, device=log_F.device)

    if obs_mask.any():
        tb = t_bin[obs_mask]
        k  = (cause[obs_mask] - 1).long()
        loss[obs_mask] = -log_F[obs_mask, tb, k]

    if cens_mask.any():
        S = overall_survival(log_F)         # (B, T)
        tb = t_bin[cens_mask].clamp(0, S.size(1) - 1)
        s_t = S[cens_mask, tb].clamp(min=1e-7)
        loss[cens_mask] = -s_t.log()

    return loss.mean()


def ranking_loss(log_F, t_bin, cause, sigma=0.1):
    """
    Simplified ranking loss: for each cause k, for pairs (i,j) where
    t_i < t_j and cause_i == k, penalise if CIF_k(t_i) < CIF_k(t_j).
    Uses batch-level subsampling for efficiency.
    """
    rl = torch.tensor(0.0, device=log_F.device)
    n_pairs = 0
    for k in range(log_F.size(2)):
        mask_k = (cause == k + 1)
        if mask_k.sum() < 2:
            continue
        idx_k  = mask_k.nonzero(as_tuple=True)[0]
        cif_k  = cif(log_F, k)                      # (B, T)
        horizon= min(log_F.size(1) - 1, t_bin.max().item())
        risk_k = cif_k[:, int(horizon)]              # (B,) scalar risk score

        t_k = t_bin[idx_k].float()
        r_k = risk_k[idx_k]
        # All pairs within this cause
        ti = t_k.unsqueeze(1); tj = t_k.unsqueeze(0)
        ri = r_k.unsqueeze(1); rj = r_k.unsqueeze(0)
        pair_mask = ti < tj
        if pair_mask.any():
            diff  = ri - rj                          # want ri > rj when ti < tj
            rl   += torch.exp(-diff[pair_mask] / sigma).mean()
            n_pairs += pair_mask.sum().item()

    return rl / max(n_pairs, 1)


# ── Training ──────────────────────────────────────────────────────────────────

def ctd_score(risk_scores, t_event, cause, target_cause):
    """Time-dependent concordance for competing risks (simplified)."""
    risk = risk_scores
    n_concordant = n_pairs = 0
    idx = np.where(cause == target_cause)[0]
    for i in idx:
        j_mask = t_event > t_event[i]
        if j_mask.sum() == 0:
            continue
        n_pairs     += j_mask.sum()
        n_concordant += (risk[i] > risk[j_mask]).sum()
    return n_concordant / n_pairs if n_pairs > 0 else 0.5


def main():
    print("=== UADT Field — 07: DeepHit Competing Risks ===")

    df, cuts, age_mean, age_std = build_competing_risks()

    feat_cols = (["age_norm", "sex_bin"] +
                 [f"site_{s}" for s in FIELD_SITES])
    X  = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    T  = torch.tensor(df["t_bin"].values,   dtype=torch.long).clamp(0, N_TIMES - 1)
    C  = torch.tensor(df["cause"].values,   dtype=torch.long)

    n_val   = int(len(X) * VAL_FRAC)
    ds      = TensorDataset(X, T, C)
    ds_tr, ds_val = random_split(ds, [len(X) - n_val, n_val],
                                  generator=torch.Generator().manual_seed(SEED))
    dl_tr  = DataLoader(ds_tr,  batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    in_dim = X.shape[1]
    model  = DeepHit(in_dim, N_TIMES, N_CAUSES, HIDDEN, CAUSE_DIM)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    best_val, patience_ctr = float("inf"), 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr = 0.0
        for xb, tb, cb in dl_tr:
            opt.zero_grad()
            log_F = model(xb)
            loss  = (ALPHA * nll_loss(log_F, tb, cb) +
                     (1 - ALPHA) * ranking_loss(log_F, tb, cb, SIGMA))
            loss.backward(); opt.step()
            tr += loss.item()

        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, tb, cb in dl_val:
                log_F = model(xb)
                vl += (ALPHA * nll_loss(log_F, tb, cb) +
                       (1 - ALPHA) * ranking_loss(log_F, tb, cb, SIGMA)).item()
        tr /= len(dl_tr); vl /= len(dl_val)
        sched.step(vl)
        history.append({"epoch": epoch, "train": tr, "val": vl})

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={tr:.4f} val={vl:.4f}")

        if vl < best_val - 1e-5:
            best_val, patience_ctr = vl, 0
            torch.save(model.state_dict(), BASE / "data/deephit_weights.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop epoch {epoch}"); break

    model.load_state_dict(torch.load(BASE / "data/deephit_weights.pt"))
    model.eval()

    with torch.no_grad():
        log_F_all = model(X)
        cif1_all  = cif(log_F_all, 0).numpy()   # (N, T) CIF for death
        cif2_all  = cif(log_F_all, 1).numpy()   # (N, T) CIF for 2nd cancer

    t_np = df["t_event"].values
    c_np = df["cause"].values

    # ── C-td evaluation ───────────────────────────────────────────────────────
    # Use CIF at last time bin as scalar risk score
    ctd1 = ctd_score(cif1_all[:, -1], t_np, c_np, target_cause=1)
    ctd2 = ctd_score(cif2_all[:, -1], t_np, c_np, target_cause=2)
    print(f"\n  C-td (death):         {ctd1:.3f}")
    print(f"  C-td (2nd UADT ca.):  {ctd2:.3f}")
    print(f"  (Cox C-index from 05b: ~0.60 benchmark)")

    # Time axis in years
    bin_midpoints = (cuts[:-1] + cuts[1:]) / 2 / 365.25

    # ── CIF by subgroup ───────────────────────────────────────────────────────
    age_med = df["age_first"].median()
    grp = {
        "Young M (age<55)":  ((df["age_first"] <  55) & (df["sex"] == "M")).values,
        "Old M (age≥55)":    ((df["age_first"] >= 55) & (df["sex"] == "M")).values,
        "Young F (age<55)":  ((df["age_first"] <  55) & (df["sex"] == "F")).values,
        "Old F (age≥55)":    ((df["age_first"] >= 55) & (df["sex"] == "F")).values,
    }

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#14304a","#2e7fbf","#e05c2e","#888888"]
    for ax, (cif_mat, title) in zip(axes, [
            (cif1_all, "Cause 1: Death"),
            (cif2_all, "Cause 2: New UADT cancer")]):
        for (label, mask), color in zip(grp.items(), colors):
            if mask.sum() < 5:
                continue
            mean_cif = cif_mat[mask].mean(axis=0)
            ax.plot(bin_midpoints, mean_cif, label=f"{label} (n={mask.sum():,})",
                    color=color, linewidth=2)
        ax.set_xlabel("Years from first UADT cancer")
        ax.set_ylabel("Cumulative incidence")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, bin_midpoints[-1])
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("DeepHit competing risks — CIF by age group and sex",
                 fontsize=12, color="#14304a", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cif_curves.png", dpi=150); plt.close()

    # Risk group trajectories: top vs bottom quartile of cause-1 risk
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    q75 = np.percentile(cif1_all[:, -1], 75)
    q25 = np.percentile(cif1_all[:, -1], 25)
    hi  = cif1_all[:, -1] >= q75
    lo  = cif1_all[:, -1] <= q25
    for ax, (cif_mat, title, caus) in zip(axes, [
            (cif1_all, "Cause 1: Death", 1),
            (cif2_all, "Cause 2: 2nd UADT cancer", 2)]):
        ax.plot(bin_midpoints, cif_mat[hi].mean(0), color="#e05c2e",
                label=f"High-risk quartile (n={hi.sum():,})", linewidth=2)
        ax.plot(bin_midpoints, cif_mat[lo].mean(0), color="#2e7fbf",
                label=f"Low-risk quartile (n={lo.sum():,})", linewidth=2)
        ax.fill_between(bin_midpoints,
                         cif_mat[hi].mean(0) - cif_mat[hi].std(0),
                         cif_mat[hi].mean(0) + cif_mat[hi].std(0),
                         alpha=0.15, color="#e05c2e")
        ax.fill_between(bin_midpoints,
                         cif_mat[lo].mean(0) - cif_mat[lo].std(0),
                         cif_mat[lo].mean(0) + cif_mat[lo].std(0),
                         alpha=0.15, color="#2e7fbf")
        ax.set_xlabel("Years"); ax.set_ylabel("CIF")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle("DeepHit — risk group separation\n(quartiles of predicted 3-yr death risk)",
                 fontsize=12, color="#14304a", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_risk_groups.png", dpi=150); plt.close()

    # Covariate effect: age sweep for C15 male
    ages_sweep  = np.arange(40, 81, 5)
    c15_idx     = FIELD_SITES.index("C15")
    cif1_by_age = []
    cif2_by_age = []
    with torch.no_grad():
        for ag in ages_sweep:
            x = np.zeros(in_dim, dtype=np.float32)
            x[0] = (ag - age_mean) / age_std    # age_norm
            x[1] = 1.0                           # sex = M
            x[2 + c15_idx] = 1.0                # first site = C15
            log_F_a = model(torch.tensor(x).unsqueeze(0))
            cif1_by_age.append(cif(log_F_a, 0).squeeze().numpy())
            cif2_by_age.append(cif(log_F_a, 1).squeeze().numpy())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.plasma
    for ax, (cif_list, title) in zip(axes, [
            (cif1_by_age, "Cause 1: Death"),
            (cif2_by_age, "Cause 2: 2nd UADT cancer")]):
        for i, (ag, c) in enumerate(zip(ages_sweep, cif_list)):
            color = cmap(i / len(ages_sweep))
            ax.plot(bin_midpoints, c, color=color, label=f"Age {ag}", linewidth=1.5)
        ax.set_xlabel("Years"); ax.set_ylabel("CIF")
        ax.set_title(f"{title}\n(first site = C15 esophagus, male)", fontsize=10)
        ax.set_ylim(0, 1); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.suptitle("Age modulation of competing risks — C15 male patients",
                 fontsize=12, color="#14304a", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_covariate_effect.png", dpi=150); plt.close()

    # Summary CSV
    horizons_days = [h * 365.25 for h in EVAL_HORIZONS_YR]
    horizon_bins  = [int(np.digitize(hd, cuts[1:])) for hd in horizons_days]
    horizon_bins  = [min(b, N_TIMES - 1) for b in horizon_bins]

    summary_rows = []
    for yr, tb in zip(EVAL_HORIZONS_YR, horizon_bins):
        mean_cif1 = float(cif1_all[:, tb].mean())
        mean_cif2 = float(cif2_all[:, tb].mean())
        summary_rows.append({
            "horizon_yr": yr,
            "mean_CIF_death":    round(mean_cif1, 4),
            "mean_CIF_2nd_uadt": round(mean_cif2, 4),
        })
    sum_df = pd.DataFrame(summary_rows)
    sum_df["ctd_death"]    = round(ctd1, 4)
    sum_df["ctd_2nd_uadt"] = round(ctd2, 4)
    sum_df["cox_cindex_05b"] = 0.60   # from script 05b benchmark
    sum_df.to_csv(OUT / "deephit_summary.csv", index=False)

    print("\n  CIF at horizons (population average):")
    print(sum_df[["horizon_yr","mean_CIF_death","mean_CIF_2nd_uadt"]].to_string(index=False))
    print(f"\n  Figures → results/07_deephit/")


if __name__ == "__main__":
    main()
