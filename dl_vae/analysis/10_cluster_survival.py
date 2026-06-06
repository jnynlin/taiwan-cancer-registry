"""
Registry DL — Script 10: Cluster Survival Analysis

Tests whether VAE-derived cancer co-occurrence clusters independently predict
all-cause mortality after adjustment for age and sex.

Analysis:
  1. Silhouette analysis k=2..8 (justify k=5 choice)
  2. KMeans k=5 (seed=42, n_init=20 — identical to Script 03)
  3. Kaplan-Meier curves by cluster with global log-rank test
  4. Cox PH: cluster dummies (ref=largest) + age_first + is_male

Survival time: (end_fu) - (diag_yr-01-01), clipped to ≥1 day.
This is conservative (uses Jan 1 of diagnosis year); all patients treated consistently.

Outputs:
  results/10_cluster_survival/silhouette_scores.csv
  results/10_cluster_survival/cluster_survival_table.csv
  results/10_cluster_survival/cox_cluster_results.csv
  results/10_cluster_survival/fig_silhouette.png
  results/10_cluster_survival/fig_km_clusters.png
  results/10_cluster_survival/fig_cox_hr.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import multivariate_logrank_test
    LIFELINES_OK = True
except ImportError:
    LIFELINES_OK = False
    print("  lifelines not found; using manual KM fallback")

BASE      = Path(__file__).parent.parent
DOUT      = BASE / "data"
R03       = BASE / "results/03_latent"
OUT       = BASE / "results/10_cluster_survival"
OUT.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 5
SEED       = 42
K_RANGE    = range(2, 9)
PALETTE    = ["#2e7fbf", "#e05c2e", "#2ca02c", "#9467bd", "#8c564b"]


def approx_survival_days(meta):
    dx = pd.to_datetime(meta["diag_yr"].astype(str) + "-01-01")
    fu = pd.to_datetime(meta["end_fu"])
    return (fu - dx).dt.days.clip(lower=1)


def km_manual(ax, times, events, label, color):
    """Nelson-Aalen step KM without CI — fallback when lifelines absent."""
    t = np.sort(np.unique(times))
    s = 1.0
    curve = [(0, 1.0)]
    for ti in t:
        mask = times == ti
        n_risk = (times >= ti).sum()
        n_event = events[mask].sum()
        if n_risk > 0:
            s *= (1 - n_event / n_risk)
        curve.append((ti, s))
    xs, ys = zip(*curve)
    ax.step(np.array(xs) / 365.25, ys, where="post", color=color, lw=2, label=label)


def main():
    print("=== Registry DL — 10: Cluster Survival Analysis ===")

    mu   = np.load(DOUT / "latent_mu.npy")
    meta = pd.read_csv(DOUT / "patient_meta.csv", index_col="pid")
    N    = len(meta)
    print(f"  Loaded: {N:,} patients")

    scaler = StandardScaler()
    mu_sc  = scaler.fit_transform(mu)

    # ── 1. Silhouette analysis ───────────────────────────────────────────────
    print("  Silhouette k=2..8 (subsample 5k)…")
    rng = np.random.RandomState(SEED)
    idx = rng.choice(N, min(5000, N), replace=False)
    sil_rows = []
    for k in K_RANGE:
        km  = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        lab = km.fit_predict(mu_sc)
        sil = silhouette_score(mu_sc[idx], lab[idx])
        sil_rows.append({"k": k, "silhouette": sil})
        print(f"    k={k}: {sil:.4f}")
    sil_df = pd.DataFrame(sil_rows)
    sil_df.to_csv(OUT / "silhouette_scores.csv", index=False)

    # ── 2. KMeans k=5 ───────────────────────────────────────────────────────
    km     = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=20)
    labels = km.fit_predict(mu_sc)

    try:
        clust_df   = pd.read_csv(R03 / "cluster_profiles.csv")
        clust_name = {int(r["cluster"]): r["name"] for _, r in clust_df.iterrows()}
    except FileNotFoundError:
        clust_name = {k: f"Cluster {k}" for k in range(N_CLUSTERS)}

    # ── 3. Survival table ────────────────────────────────────────────────────
    surv = meta[["age_first", "sex", "dead", "diag_yr", "end_fu"]].copy()
    surv["cluster"]      = labels
    surv["cluster_name"] = [clust_name.get(k, f"C{k}") for k in labels]
    surv["duration"]     = approx_survival_days(surv)
    surv["is_male"]      = (surv["sex"] == "M").astype(int)
    surv = surv[surv["duration"] > 0]
    surv.to_csv(OUT / "cluster_survival_table.csv")

    ref_k = surv["cluster"].value_counts().idxmax()
    print(f"  Reference cluster: C{ref_k} [{clust_name.get(ref_k,'?')}] (largest)")
    for k in range(N_CLUSTERS):
        m = surv[surv["cluster"] == k]
        print(f"  C{k} [{clust_name.get(k,'?')}]: n={len(m):,}  "
              f"dead={int(m['dead'].sum())}  median={m['duration'].median()/365.25:.1f}yr")

    # ── Fig A: Silhouette ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sil_df["k"], sil_df["silhouette"], marker="o", color="#2e7fbf", lw=2)
    ax.axvline(N_CLUSTERS, color="#e05c2e", lw=1.5, ls="--",
               label=f"k={N_CLUSTERS} (selected)")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette analysis — VAE latent space")
    ax.set_xticks(list(K_RANGE))
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_silhouette.png", dpi=150)
    plt.close()

    # ── Fig B: KM curves ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    if LIFELINES_OK:
        res = multivariate_logrank_test(surv["duration"], surv["cluster"], surv["dead"])
        p_lr = res.p_value
        for k in range(N_CLUSTERS):
            m = surv[surv["cluster"] == k]
            kmf = KaplanMeierFitter()
            kmf.fit(m["duration"] / 365.25, m["dead"],
                    label=f"C{k}: {clust_name.get(k,'?')} (n={len(m):,})")
            kmf.plot_survival_function(ax=ax, ci_show=True,
                                       color=PALETTE[k], linewidth=2, alpha=0.85)
        ax.set_title(f"Kaplan–Meier by VAE cluster  (global log-rank p={p_lr:.2e})")
    else:
        for k in range(N_CLUSTERS):
            m = surv[surv["cluster"] == k]
            km_manual(ax, m["duration"].values, m["dead"].values,
                      label=f"C{k}: {clust_name.get(k,'?')} (n={len(m):,})",
                      color=PALETTE[k])
        ax.set_title("Kaplan–Meier by VAE cluster (manual fallback)")
    ax.set_xlabel("Years from first diagnosis")
    ax.set_ylabel("Overall survival probability")
    ax.set_xlim(0); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_km_clusters.png", dpi=150)
    plt.close()

    # ── Fig C: Cox HR + Schoenfeld PH test + time-split if violated ──────────
    if LIFELINES_OK:
        from lifelines.statistics import proportional_hazard_test

        cox_df = surv[["duration", "dead", "age_first", "is_male", "cluster"]].copy()
        for k in range(N_CLUSTERS):
            if k != ref_k:
                cox_df[f"C{k}_vs_C{ref_k}"] = (surv["cluster"] == k).astype(int)

        dummy_cols = [c for c in cox_df.columns if c.startswith("C") and "vs" in c]
        fit_cols   = ["duration", "dead", "age_first", "is_male"] + dummy_cols

        cph = CoxPHFitter()
        cph.fit(cox_df[fit_cols], duration_col="duration", event_col="dead",
                show_progress=False)
        cph.summary.to_csv(OUT / "cox_cluster_results.csv")

        cluster_rows = cph.summary[[
            "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"
        ]].loc[dummy_cols]

        print("\n  Cox cluster HRs (vs C" + str(ref_k) + "):")
        print(cluster_rows.to_string())

        # ── Schoenfeld PH test ───────────────────────────────────────────────
        print("\n  Schoenfeld proportional-hazards test:")
        ph = proportional_hazard_test(cph, cox_df[fit_cols], time_transform="rank")
        ph_df = ph.summary.copy()
        ph_df["PH_OK"] = ph_df["p"] >= 0.05
        ph_df.to_csv(OUT / "ph_test_results.csv")
        print(ph_df[["test_statistic", "p", "PH_OK"]].to_string())

        violated = ph_df[~ph_df["PH_OK"]].index.tolist()
        print(f"\n  PH violations: {violated if violated else 'none'}")

        # ── Time-split Cox at 2yr landmark if any violation ──────────────────
        SPLIT_YR   = 2.0
        SPLIT_DAYS = SPLIT_YR * 365.25

        if violated:
            print(f"\n  Running time-split Cox (landmark={SPLIT_YR}yr)…")
            # Early period: 0 → SPLIT_DAYS
            early = cox_df[cox_df["duration"] > 0].copy()
            early["duration_e"] = early["duration"].clip(upper=SPLIT_DAYS)
            early["dead_e"]     = ((early["dead"] == 1) &
                                   (early["duration"] <= SPLIT_DAYS)).astype(int)
            early = early[early["duration_e"] > 0]

            # Late period: SPLIT_DAYS → end (only patients who survived past split)
            late = cox_df[cox_df["duration"] > SPLIT_DAYS].copy()
            late["duration_l"] = late["duration"] - SPLIT_DAYS

            cph_early = CoxPHFitter()
            cph_late  = CoxPHFitter()

            fit_e = ["duration_e","dead_e","age_first","is_male"] + dummy_cols
            fit_l = ["duration_l","dead","age_first","is_male"]   + dummy_cols

            cph_early.fit(early[fit_e], duration_col="duration_e", event_col="dead_e",
                          show_progress=False)
            cph_late.fit(late[fit_l],   duration_col="duration_l", event_col="dead",
                         show_progress=False)

            early_rows = cph_early.summary[
                ["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]
            ].loc[dummy_cols]
            late_rows  = cph_late.summary[
                ["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]
            ].loc[dummy_cols]

            split_df = pd.DataFrame({
                "HR_early":    early_rows["exp(coef)"].round(3),
                "CI_lo_early": early_rows["exp(coef) lower 95%"].round(3),
                "CI_hi_early": early_rows["exp(coef) upper 95%"].round(3),
                "p_early":     early_rows["p"].round(6),
                "HR_late":     late_rows["exp(coef)"].round(3),
                "CI_lo_late":  late_rows["exp(coef) lower 95%"].round(3),
                "CI_hi_late":  late_rows["exp(coef) upper 95%"].round(3),
                "p_late":      late_rows["p"].round(6),
            })
            split_df.to_csv(OUT / "cox_timesplit_results.csv")
            print(f"\n  Time-split Cox (early ≤{SPLIT_YR}yr | late >{SPLIT_YR}yr):")
            print(split_df.to_string())

            # Figure: side-by-side early/late forest plot
            fig, axes = plt.subplots(1, 2, figsize=(14, max(3, len(dummy_cols)*0.9+2)),
                                     sharey=True)
            for ax, rows, period in zip(axes,
                                        [early_rows, late_rows],
                                        [f"Early (≤{SPLIT_YR}yr)", f"Late (>{SPLIT_YR}yr)"]):
                y   = range(len(dummy_cols))
                hrs = rows["exp(coef)"].values
                lo  = rows["exp(coef) lower 95%"].values
                hi  = rows["exp(coef) upper 95%"].values
                ps  = rows["p"].values
                ax.scatter(hrs, list(y), color="#2e7fbf", zorder=5, s=60)
                ax.hlines(list(y), lo, hi, color="#2e7fbf", lw=2)
                ax.axvline(1.0, color="gray", lw=1, ls="--")
                ax.set_yticks(list(y))
                ax.set_yticklabels(dummy_cols, fontsize=9)
                ax.set_xlabel("Hazard ratio (95% CI)")
                ax.set_title(f"{period}")
                for i, (hr, hi_i, p) in enumerate(zip(hrs, hi, ps)):
                    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
                    ax.text(hi_i * 1.05, i, f"{hr:.2f}{sig}", va="center", fontsize=8)
            fig.suptitle(
                f"Time-split Cox — landmark {SPLIT_YR}yr (PH violated: {', '.join(violated)})\n"
                f"Adjusted for age + sex  |  ref=C{ref_k}", fontsize=11)
            fig.tight_layout()
            fig.savefig(OUT / "fig_cox_timesplit.png", dpi=150)
            plt.close()
            print(f"  Saved: fig_cox_timesplit.png")

        # ── Standard forest plot (full follow-up) ────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, max(3, len(dummy_cols)*0.9+2)),
                                 gridspec_kw={"width_ratios": [2, 1]})
        ax = axes[0]
        y   = range(len(dummy_cols))
        hrs = cluster_rows["exp(coef)"].values
        lo  = cluster_rows["exp(coef) lower 95%"].values
        hi  = cluster_rows["exp(coef) upper 95%"].values
        ps  = cluster_rows["p"].values
        ax.scatter(hrs, list(y), color="#2e7fbf", zorder=5, s=60)
        ax.hlines(list(y), lo, hi, color="#2e7fbf", lw=2)
        ax.axvline(1.0, color="gray", lw=1, ls="--")
        ax.set_yticks(list(y))
        ax.set_yticklabels(dummy_cols, fontsize=9)
        ax.set_xlabel("Hazard ratio (95% CI) — Cox adjusted for age + sex")
        ax.set_title(f"Full follow-up  |  ref=C{ref_k}")
        for i, (hr, hi_i, p) in enumerate(zip(hrs, hi, ps)):
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            ax.text(hi_i * 1.05, i, f"{hr:.2f}{sig}", va="center", fontsize=8)

        # PH test summary panel
        ax2 = axes[1]
        ax2.axis("off")
        ph_display = ph_df[["test_statistic","p","PH_OK"]].copy()
        ph_display.index = [i.replace("_vs_", " vs ") for i in ph_display.index]
        ph_display["test_statistic"] = ph_display["test_statistic"].map(lambda x: f"{x:.1f}")
        ph_display["p"]    = ph_display["p"].map(lambda x: f"{x:.2e}")
        ph_display["PH_OK"] = ph_display["PH_OK"].map({True: "✓", False: "⚠"})
        col_labels = ["χ²", "p", "OK"]
        cell_text  = [[str(v) for v in row] for row in ph_display.values.tolist()]
        row_labels = list(ph_display.index)
        t = ax2.table(cellText=cell_text, rowLabels=row_labels,
                      colLabels=col_labels, loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(8)
        for (r, c), cell in t.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white")
            elif "FAIL" in str(cell.get_text().get_text()):
                cell.set_facecolor("#FDECEA")
            elif "✓" in str(cell.get_text().get_text()):
                cell.set_facecolor("#E8F5E9")
        ax2.set_title("Schoenfeld PH test", fontsize=9, pad=4)

        fig.suptitle(
            f"Cluster survival HRs vs C{ref_k} (ref)\n"
            f"PH {'satisfied — full-FU HRs valid' if not violated else 'violated for: ' + str(violated)}",
            fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "fig_cox_hr.png", dpi=150)
        plt.close()
    else:
        print("  lifelines absent — Cox HR figure skipped")

    print(f"  Saved → {OUT}/")


if __name__ == "__main__":
    main()
