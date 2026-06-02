"""
Registry DL — Script 06: Hereditary Syndrome Screening

Matches anomalous patients (top 1% by autoencoder reconstruction error)
against known hereditary cancer syndrome site patterns.

Syndromes screened (ICD-O C-codes, restricted to sites present in registry):
  Lynch     : C18/C20 colorectal + C54 endometrial + C56 ovarian + C16 gastric + C67 bladder
  BRCA      : C50 breast + C56 ovarian + C61 prostate + C25 pancreatic
  MEN1      : C25 pancreatic + C37 thymus (pituitary not in registry)
  Li-Fraumeni: C50 breast + C71 brain + C74 adrenal + C49 soft tissue
  FAP       : C18/C20 colorectal + C16 gastric
  VHL       : C64 renal + C71 brain/CNS
  Cowden    : C50 breast + C73 thyroid + C54 endometrial

Match score = n_syndrome_sites_present / n_syndrome_sites_total (Jaccard-like)
Candidates require: match_score ≥ 0.40 AND n_sites ≥ 2 AND anomaly_rank ≤ top 1%

Outputs:
  results/06_syndrome/syndrome_candidates.csv
  results/06_syndrome/fig_syndrome_counts.png
  results/06_syndrome/fig_candidate_profiles.png
  results/Syndrome_Screen_Draft.pdf
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE  = Path(__file__).parent.parent
DOUT  = BASE / "data"
OUT   = BASE / "results/06_syndrome"
R05   = BASE / "results/05_anomaly"
OUT.mkdir(parents=True, exist_ok=True)

NAVY   = "#14304a"
ACCENT = "#2e7fbf"

# Hereditary syndrome site patterns (ICD-O C-codes)
SYNDROMES = {
    "Lynch":        {"C18","C20","C54","C56","C16","C67"},
    "BRCA":         {"C50","C56","C61","C25"},
    "MEN1":         {"C25","C37","C18"},
    "Li-Fraumeni":  {"C50","C71","C74","C49"},
    "FAP":          {"C18","C20","C16"},
    "VHL":          {"C64","C71"},
    "Cowden":       {"C50","C73","C54"},
}

MATCH_THRESHOLD = 0.40
MIN_SITES       = 2
TOP_FRAC        = 0.01


def match_score(patient_sites, syndrome_sites):
    """Fraction of syndrome-defining sites present in patient."""
    present = patient_sites & syndrome_sites
    return len(present) / len(syndrome_sites) if syndrome_sites else 0.0


def main():
    print("=== Registry DL — 06: Hereditary Syndrome Screening ===")

    scores  = pd.read_csv(R05  / "anomaly_scores.csv")
    X_df    = pd.read_csv(DOUT / "cancer_matrix.csv", index_col="pid")
    meta    = pd.read_csv(DOUT / "patient_meta.csv",  index_col="pid")
    sites_in_registry = set(X_df.columns.tolist())

    # Restrict syndrome definitions to sites actually in registry
    syndromes_effective = {
        name: s & sites_in_registry for name, s in SYNDROMES.items()
    }
    print("  Effective syndrome sites (restricted to registry):")
    for name, s in syndromes_effective.items():
        print(f"    {name}: {sorted(s)}")

    # Flagged patients
    n_flag  = int(len(scores) * TOP_FRAC)
    flagged = scores.head(n_flag).copy()
    flagged["patient_sites"] = flagged["sites"].apply(
        lambda x: set(str(x).split("+")) if pd.notna(x) and x else set())

    # Score each flagged patient against each syndrome
    cand_rows = []
    for _, row in flagged.iterrows():
        pt_sites = row["patient_sites"]
        if len(pt_sites) < MIN_SITES:
            continue
        best_score, best_syndrome, best_hits = 0.0, None, set()
        all_scores = {}
        for name, syn_sites in syndromes_effective.items():
            if not syn_sites:
                continue
            ms = match_score(pt_sites, syn_sites)
            all_scores[name] = ms
            if ms > best_score:
                best_score    = ms
                best_syndrome = name
                best_hits     = pt_sites & syn_sites

        if best_score >= MATCH_THRESHOLD and best_syndrome:
            demo = meta.loc[row["pid"]] if row["pid"] in meta.index else None
            cand_rows.append({
                "pid":             row["pid"],
                "anomaly_rank":    row["rank"],
                "anomaly_score":   row["anomaly_score"],
                "n_sites":         row["n_sites"],
                "sites":           row["sites"],
                "best_syndrome":   best_syndrome,
                "match_score":     round(best_score, 3),
                "matching_sites":  "+".join(sorted(best_hits)),
                "age_first":       demo["age_first"] if demo is not None else None,
                "sex":             demo["sex"]       if demo is not None else None,
                "dead":            demo["dead"]       if demo is not None else None,
                **{f"score_{k}": round(v, 3) for k, v in all_scores.items()},
            })

    cand_df = pd.DataFrame(cand_rows).sort_values(
        ["best_syndrome", "match_score"], ascending=[True, False])
    cand_df.to_csv(OUT / "syndrome_candidates.csv", index=False)

    print(f"\n  Candidates (match≥{MATCH_THRESHOLD}, n_sites≥{MIN_SITES}): {len(cand_df)}")
    for syn in SYNDROMES:
        n = (cand_df["best_syndrome"] == syn).sum() if len(cand_df) else 0
        print(f"    {syn}: {n}")

    # ── Figures ───────────────────────────────────────────────────────────────

    # Fig A: syndrome candidate counts
    syn_counts = cand_df["best_syndrome"].value_counts() if len(cand_df) else pd.Series(dtype=int)
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(syn_counts):
        ax.bar(syn_counts.index, syn_counts.values, color=ACCENT)
        for i, (k, v) in enumerate(syn_counts.items()):
            ax.text(i, v + 0.3, str(v), ha="center", fontsize=10)
    ax.set_ylabel("Candidate patients")
    ax.set_title(f"Hereditary syndrome candidates\n"
                 f"(top {TOP_FRAC*100:.0f}% anomalous, match≥{MATCH_THRESHOLD})")
    fig.tight_layout()
    fig.savefig(OUT / "fig_syndrome_counts.png", dpi=150); plt.close()

    # Fig B: candidate profile heatmap (up to top 40 candidates)
    show = cand_df.head(40)
    all_sites = sorted({s for row in show["sites"] if pd.notna(row)
                        for s in str(row).split("+") if s})
    if len(show) and len(all_sites):
        mat = np.zeros((len(show), len(all_sites)))
        for i, (_, row) in enumerate(show.iterrows()):
            pt_s = set(str(row["sites"]).split("+")) if pd.notna(row["sites"]) else set()
            for j, s in enumerate(all_sites):
                mat[i, j] = 1 if s in pt_s else 0

        fig, ax = plt.subplots(figsize=(max(8, len(all_sites) * 0.5), max(6, len(show) * 0.25)))
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(all_sites))); ax.set_xticklabels(all_sites, rotation=90, fontsize=8)
        ax.set_yticks(range(len(show)))
        labels = [f"#{r['anomaly_rank']} {r['best_syndrome']} ({r['match_score']:.2f})"
                  for _, r in show.iterrows()]
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"Top {len(show)} syndrome candidate profiles\n"
                     "(blue = cancer site present)", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / "fig_candidate_profiles.png", dpi=150); plt.close()

    # ── Draft PDF ─────────────────────────────────────────────────────────────
    pdf_path = BASE / "results/Syndrome_Screen_Draft.pdf"

    def footer(fig, page, total=4):
        fig.text(0.5, 0.01,
                 f"Taiwan Cancer Registry — Hereditary Syndrome Screen  |  "
                 f"Page {page}/{total}  |  Draft 2026-06-02",
                 ha="center", fontsize=7, color="#888888")

    def img(ax, path, title=None):
        p = Path(path)
        if p.exists():
            from matplotlib.image import imread
            ax.imshow(imread(str(p)), aspect="equal"); ax.axis("off")
            if title: ax.set_title(title, fontsize=9, pad=3)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"[{p.name} missing]", ha="center", va="center",
                    transform=ax.transAxes, color="red")

    with PdfPages(str(pdf_path)) as pdf:

        # Page 1: Title + overview
        fig = plt.figure(figsize=(11, 8.5))
        ax_t = fig.add_axes([0.05, 0.55, 0.90, 0.38])
        ax_t.set_facecolor(NAVY); ax_t.axis("off")
        ax_t.text(0.5, 0.65, "Hereditary Cancer Syndrome Screening",
                  ha="center", va="center", fontsize=18, color="white",
                  fontweight="bold", transform=ax_t.transAxes)
        ax_t.text(0.5, 0.30,
                  "Autoencoder anomaly detection · Taiwan Cancer Registry\n"
                  "78,442 patients · 37 sites · 7 syndrome patterns",
                  ha="center", va="center", fontsize=12, color="#aaccee",
                  transform=ax_t.transAxes)
        ax_s = fig.add_axes([0.05, 0.10, 0.90, 0.38])
        ax_s.axis("off")
        syn_str = "\n".join(f"  {n}: {sorted(s)}" for n, s in syndromes_effective.items() if s)
        stats = (
            f"Method: Bottleneck autoencoder (dim=16) trained on all patients.\n"
            f"Anomaly score = per-patient sum-BCE. Top 1% ({n_flag:,} patients) screened.\n"
            f"Match threshold ≥ {MATCH_THRESHOLD} of syndrome-defining sites present.\n\n"
            f"Total candidates: {len(cand_df)}\n\n"
            f"Syndrome definitions (effective sites in registry):\n{syn_str}"
        )
        ax_s.text(0.5, 0.92, stats, ha="center", va="top", fontsize=9,
                  transform=ax_s.transAxes, color=NAVY,
                  bbox=dict(facecolor="#f0f4f8", edgecolor=ACCENT, boxstyle="round,pad=0.5"))
        footer(fig, 1); pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # Page 2: Score distribution + candidate counts
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        img(axes[0], R05 / "fig_score_dist.png", "Anomaly score distribution (top 1% flagged)")
        img(axes[1], OUT / "fig_syndrome_counts.png", "Candidates by syndrome pattern")
        fig.suptitle("Autoencoder Anomaly Scores + Syndrome Candidates",
                     fontsize=13, color=NAVY, fontweight="bold")
        footer(fig, 2); fig.tight_layout(rect=[0,0.03,1,0.95])
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # Page 3: Candidate profile heatmap
        fig, ax = plt.subplots(figsize=(11, 8))
        img(ax, OUT / "fig_candidate_profiles.png",
            "Top 40 candidate profiles — cancer sites present (blue)\n"
            "ranked by anomaly score; labeled with best-matching syndrome")
        fig.suptitle("Syndrome Candidate Profiles", fontsize=13, color=NAVY, fontweight="bold")
        footer(fig, 3); pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # Page 4: Candidate table + limitations
        fig = plt.figure(figsize=(11, 8.5))
        ax_tbl = fig.add_axes([0.02, 0.35, 0.96, 0.55])
        ax_tbl.axis("off")
        show_tbl = cand_df[["pid","anomaly_rank","n_sites","sites",
                             "best_syndrome","match_score","matching_sites",
                             "age_first","sex"]].head(25)
        show_tbl.columns = ["PID","Rank","N sites","Sites","Best syndrome",
                             "Match","Matching sites","Age","Sex"]
        if len(show_tbl):
            tbl = ax_tbl.table(cellText=show_tbl.values, colLabels=show_tbl.columns,
                               loc="center", cellLoc="left")
            tbl.auto_set_font_size(False); tbl.set_fontsize(6.5)
            tbl.scale(1, 1.3)
        ax_tbl.set_title(f"Top syndrome candidates (showing up to 25 of {len(cand_df)})",
                         fontsize=10, pad=6)

        ax_lim = fig.add_axes([0.02, 0.02, 0.96, 0.28])
        ax_lim.axis("off")
        lim = (
            "Limitations & caveats:\n"
            "① Syndrome definitions use only the 37 registry sites — pituitary (MEN1), "
            "sarcoma (Li-Fraumeni), and renal (VHL) are under-represented or absent; match scores are underestimates.\n"
            "② High anomaly score reflects deviation from the registry average, not necessarily hereditary etiology — "
            "rare sporadic co-occurrences will also appear in the tail.\n"
            "③ No age-of-onset filter applied — hereditary syndromes present earlier; "
            "filtering to age_first < 50 would improve specificity.\n"
            "④ Registry lacks family history data — confirmation requires genetic counselling referral.\n"
            "⑤ Match threshold (≥0.40) is arbitrary; clinical follow-up should lower threshold to ≥0.33 (2/6 sites)."
        )
        ax_lim.text(0.02, 0.95, lim, transform=ax_lim.transAxes, fontsize=8.5,
                    va="top", color=NAVY)
        fig.suptitle("Candidate Table + Limitations", fontsize=13, color=NAVY, fontweight="bold")
        footer(fig, 4); pdf.savefig(fig, bbox_inches="tight"); plt.close()

    print(f"  Draft PDF → {pdf_path}")


if __name__ == "__main__":
    main()
