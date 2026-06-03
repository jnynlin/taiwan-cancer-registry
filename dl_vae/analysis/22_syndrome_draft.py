"""
Registry DL — Script 22: Hereditary Syndrome Validation Draft PDF

Findings summary:
  Raw screen (Script 06):  128 candidates; 43 actionable (LFS=26, Cowden=17)
  After refinement:        3 high-confidence Cowden (C50+C54, match_score=1.0)
                           0 high-confidence LFS (none have C71 brain tumour)

  Principal message: Registry-based syndrome screening is feasible for Cowden
  (breast + endometrial well-captured) but severely limited for LFS (brain
  tumours present in registry but absent from all 26 LFS candidates; adrenal
  C74 and soft tissue C49 absent entirely).

Output:
  results/Syndrome_Validation_Draft.pdf
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = Path(__file__).parent.parent
R06  = BASE / "results/06_syndrome"
R21  = BASE / "results/21_syndrome"
OUT  = BASE / "results/Syndrome_Validation_Draft.pdf"

NAVY   = "#14304a"
ACCENT = "#2e7fbf"
PURPLE = "#9467bd"
RED    = "#d62728"


def flow(ax, text, x=0.05, y=0.95, fs=9, **kw):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs,
            va="top", ha="left", wrap=True, **kw)


def footer(ax, page, n=8):
    ax.text(0.99, 0.01, f"Taiwan Cancer Registry — Syndrome Validation  |  p. {page}/{n}",
            transform=ax.transAxes, fontsize=7, color="#888888", ha="right", va="bottom")


def img(ax, path, title=None):
    from matplotlib.image import imread
    p = Path(path)
    if p.exists():
        ax.imshow(imread(str(p)))
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=8, pad=3)
    else:
        ax.text(0.5, 0.5, f"[missing: {p.name}]", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="red")
        ax.axis("off")


def main():
    print("=== Registry DL — 22: Syndrome Validation Draft PDF ===")

    cand_all  = pd.read_csv(R06 / "syndrome_candidates.csv")
    cand_ref  = pd.read_csv(R21 / "refined_candidates.csv")
    timing    = pd.read_csv(R21 / "timing_table.csv")

    n_raw         = len(cand_all)
    n_actionable  = len(cand_all[cand_all["best_syndrome"].isin(["Li-Fraumeni","Cowden"])])
    n_lfs_raw     = len(cand_all[cand_all["best_syndrome"]=="Li-Fraumeni"])
    n_cow_raw     = len(cand_all[cand_all["best_syndrome"]=="Cowden"])
    n_high        = len(cand_ref[cand_ref["refined"]==True])
    n_cow_hi      = len(cand_ref[(cand_ref["best_syndrome"]=="Cowden") & (cand_ref["refined"]==True)])

    timing_cow = timing[(timing["best_syndrome"]=="Cowden") & (timing["match_score"]==1.0)]
    timing_lfs = timing[timing["best_syndrome"]=="Li-Fraumeni"]

    with PdfPages(str(OUT)) as pdf:

        # ── Page 1: Title + executive summary ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        fig.patch.set_facecolor("white")
        ax.add_patch(plt.Rectangle((0,0.88),1,0.12,
                     transform=ax.transAxes, color=NAVY, zorder=0))
        ax.text(0.5, 0.935, "Taiwan Cancer Registry", transform=ax.transAxes,
                ha="center", fontsize=16, color="white", fontweight="bold")
        ax.text(0.5, 0.895, "Hereditary Cancer Syndrome Candidate Validation",
                transform=ax.transAxes, ha="center", fontsize=12, color="white")

        summary = (
            "Background\n\n"
            "An autoencoder trained on 78,442 patients × 37 cancer sites identified\n"
            "the top 1% anomalous multi-cancer patients (n=784) and matched them to\n"
            "seven hereditary syndrome site patterns. 43 actionable candidates were\n"
            "flagged for Li-Fraumeni (LFS) or Cowden syndrome.\n\n"
            "This report applies phenotypic refinement criteria and evaluates the\n"
            "clinical plausibility of each classification.\n\n"
            "Screening pipeline\n\n"
            f"  Autoencoder anomaly top 1%        → {cand_all['pid'].nunique():6} candidates\n"
            f"  match_score ≥ 0.40 filter         → {n_raw:6} syndrome candidates\n"
            f"  Actionable (LFS + Cowden only)    → {n_actionable:6} patients\n"
            f"    Li-Fraumeni (raw)               → {n_lfs_raw:6}\n"
            f"    Cowden (raw)                    → {n_cow_raw:6}\n\n"
            "After phenotypic refinement\n\n"
            f"  LFS refined (C71 present, age<50) →      0 ← all failed\n"
            f"  Cowden refined (match_score=1.0)  → {n_cow_hi:6} ← high-confidence\n\n"
            "Critical structural finding\n\n"
            "  LFS effective registry definition: {C50, C71} only\n"
            "  (C74 adrenal, C49 soft tissue absent from registry)\n"
            "  → All 26 LFS candidates have C50 (breast) + unrelated cancer only.\n"
            "    None have C71 (brain/CNS) — the hallmark LFS tumour.\n"
            "  → 'Li-Fraumeni' label at match_score=0.5 is clinically non-specific.\n\n"
            "  Cowden effective registry definition: {C50, C54} only\n"
            "  (C73 thyroid absent from registry)\n"
            "  → 3 patients have both C50 (breast) + C54 (endometrial) — consistent\n"
            "    with published Cowden phenotype."
        )
        ax.text(0.07, 0.82, summary, transform=ax.transAxes, fontsize=9,
                va="top", ha="left", fontfamily="monospace",
                bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", pad=8))
        footer(ax, 1)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 2: Effective syndrome definitions + match score distribution ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        img(axes[0], R21/"fig_match_score_dist.png",
            "Match score distributions — LFS vs Cowden")

        ax_r = axes[1]
        ax_r.axis("off")
        table_data = [
            ["Syndrome","Original sites","Effective (in registry)","N effective"],
            ["Lynch",      "C18/C20/C54/C56/C16/C67","C18/C20/C54/C56/C16/C67","6"],
            ["BRCA",       "C50/C56/C61/C25",         "C50/C56/C61/C25",         "4"],
            ["Li-Fraumeni","C50/C71/C74/C49",          "C50/C71",                 "2 ⚠️"],
            ["Cowden",     "C50/C73/C54",              "C50/C54",                 "2 ⚠️"],
            ["MEN1",       "C25/C37/C18",              "C25/C18 (C37 absent)",    "2 ⚠️"],
            ["FAP",        "C18/C20/C16",              "C18/C20/C16",             "3"],
            ["VHL",        "C64/C71",                  "C71 (C64 absent)",        "1 ⚠️"],
        ]
        tbl = ax_r.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc="center", loc="center",
                         bbox=[0.0, 0.3, 1.0, 0.65])
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        for (r,c), cell in tbl.get_celld().items():
            if r == 0: cell.set_facecolor(NAVY); cell.set_text_props(color="white")
            elif "⚠" in str(tbl.get_celld()[(r,3)].get_text().get_text()):
                cell.set_facecolor("#fff3cd")
        ax_r.text(0.05, 0.28,
                  "⚠️ = fewer than 3 effective sites — high false-positive risk",
                  transform=ax_r.transAxes, fontsize=8, va="top")
        ax_r.set_title("Effective syndrome site definitions after registry restriction", fontsize=9)
        fig.suptitle("Syndrome definitions — registry coverage determines specificity",
                     fontsize=11, color=NAVY)
        footer(ax_r, 2)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 3: Age distributions ──────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        img(axes[0], R21/"fig_age_first_dist.png",
            "Age at first diagnosis vs published syndrome distributions")

        ax_r = axes[1]
        # Summary interpretation panel
        ax_r.axis("off")
        interp = (
            "Age at first diagnosis — interpretation\n\n"
            "Li-Fraumeni syndrome (published):\n"
            "  Median age at first cancer ~30 yr\n"
            "  Classic criterion: sarcoma <45 yr in proband\n"
            "  Chompret criterion: ANY cancer <46 yr\n\n"
            "Our LFS candidates:\n"
            f"  N=26, median age {cand_all[cand_all['best_syndrome']=='Li-Fraumeni']['age_first'].median():.0f} yr\n"
            "  Range 37–80 yr — substantially older than published LFS\n"
            "  NONE have C71 brain tumour (hallmark site)\n"
            "  → These are breast-cancer patients with an incidental\n"
            "    second cancer, NOT clinical LFS.\n\n"
            "Cowden syndrome (published):\n"
            "  Breast onset median 38–46 yr (Pilarski 2009)\n"
            "  Endometrial onset median 35–55 yr\n\n"
            "Our Cowden candidates:\n"
            f"  N=17, median age {cand_all[cand_all['best_syndrome']=='Cowden']['age_first'].median():.0f} yr\n"
            "  High-confidence (C50+C54): ages 46, 49, 57 yr\n"
            "  → Overlaps published Cowden breast/endometrial range\n"
            "  → Clinically plausible for PTEN testing"
        )
        ax_r.text(0.05, 0.95, interp, transform=ax_r.transAxes, fontsize=9,
                  va="top",
                  bbox=dict(facecolor="#f0f4f8", edgecolor="#cccccc", pad=6))
        ax_r.set_title("Age at first diagnosis — clinical plausibility", fontsize=9)
        fig.suptitle("Phenotypic comparison — flagged candidates vs published syndrome age distributions",
                     fontsize=11, color=NAVY)
        footer(ax_r, 3)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 4: High-confidence Cowden — detailed profile ─────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        img(axes[0], R21/"fig_site_co_occurrence.png",
            "Site co-occurrence — Cowden candidates (n=17)")

        ax_r = axes[1]
        ax_r.axis("off")
        # Table: 3 high-confidence candidates
        hi = cand_ref[cand_ref["refined"]==True].copy()
        hi["priority_score"] = hi["priority_score"].round(2)

        table_rows = []
        for _, row in hi.iterrows():
            # Retrieve timing for this patient
            pt_timing = timing_cow[timing_cow["pid"].astype(str) == str(row["pid"])]
            gap_str = f"{pt_timing['gap_months'].values[0]:.0f} mo" if len(pt_timing)>0 else "—"
            table_rows.append([
                str(row["pid"])[-6:] + "…",   # anonymised tail
                row["sites"],
                f"{row['age_first']:.0f}",
                "F",
                "Dead" if row["dead"]==1 else "Alive",
                gap_str,
                f"{row['priority_score']:.1f}",
            ])
        tbl2 = ax_r.table(
            cellText=table_rows,
            colLabels=["PID (tail)","Sites","Age","Sex","Status","Gap","Priority"],
            cellLoc="center", loc="upper center",
            bbox=[0.0, 0.50, 1.0, 0.40]
        )
        tbl2.auto_set_font_size(False); tbl2.set_fontsize(9)
        for (r,c), cell in tbl2.get_celld().items():
            if r==0: cell.set_facecolor(NAVY); cell.set_text_props(color="white")
            else: cell.set_facecolor("#f0f0ff")

        note = (
            "3 Cowden high-confidence candidates\n"
            "(C50 breast + C54 endometrial, both present)\n\n"
            "Cowden syndrome (PTEN hamartoma tumour syndrome)\n"
            "OMIM #158350 — autosomal dominant PTEN mutation\n\n"
            "Clinical significance:\n"
            "  • Lifetime breast cancer risk: 85%\n"
            "  • Lifetime endometrial cancer risk: 28%\n"
            "  • Thyroid (C73, absent from this registry): 30–35%\n"
            "  • Recommendation: PTEN germline testing for patients\n"
            "    with synchronous/metachronous breast + endometrial\n\n"
            "Registry limitations:\n"
            "  • C73 thyroid absent → cannot identify thyroid arm\n"
            "  • No family history data → cannot apply full NCCN criteria\n"
            "  • 3 candidates represent minimum estimate"
        )
        ax_r.text(0.05, 0.45, note, transform=ax_r.transAxes, fontsize=8.5,
                  va="top",
                  bbox=dict(facecolor="#f5eeff", edgecolor="#9467bd", pad=6))
        ax_r.set_title(f"High-confidence Cowden candidates (n={n_cow_hi})", fontsize=9)
        fig.suptitle("Cowden syndrome — high-confidence candidate profiles",
                     fontsize=11, color=NAVY)
        footer(ax_r, 4)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 5: LFS failure analysis ──────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: LFS site distribution (what sites do LFS candidates have beyond C50?)
        lfs_cands = cand_all[cand_all["best_syndrome"]=="Li-Fraumeni"].copy()
        other_sites = []
        for _, row in lfs_cands.iterrows():
            for s in row["sites"].split("+"):
                if s != "C50":
                    other_sites.append(s)
        from collections import Counter
        site_cnt = Counter(other_sites)
        sites_sorted = sorted(site_cnt.keys(), key=lambda x: site_cnt[x], reverse=True)

        ax = axes[0]
        colors_bar = ["#d62728" if s=="C71" else "#aaaaaa" for s in sites_sorted]
        ax.barh(sites_sorted, [site_cnt[s] for s in sites_sorted],
                color=colors_bar, alpha=0.8)
        ax.set_xlabel("N LFS candidates")
        ax.set_title("Sites co-occurring with C50 in LFS candidates\n"
                     "(red = C71 brain — expected hallmark; absent from all 26)")
        if "C71" not in sites_sorted:
            ax.text(0.5, 0.05, "C71 not present in any LFS candidate",
                    transform=ax.transAxes, ha="center", fontsize=9, color="red",
                    bbox=dict(facecolor="white", edgecolor="red", pad=3))

        ax_r2 = axes[1]
        ax_r2.axis("off")
        lfs_note = (
            "Why all 26 LFS candidates fail refinement\n\n"
            "Criterion 1: C71 (brain/CNS) must be present\n"
            "  → 0 of 26 LFS candidates have C71\n"
            "  → C71 IS present in registry (n=718 patients)\n"
            "  → LFS flagging is based on C50 alone (match_score=0.5\n"
            "     = 1 of 2 effective sites)\n\n"
            "Criterion 2: age_first < 50 (Chompret-like)\n"
            "  → LFS candidates: median age 58.5 yr (range 37–80)\n"
            "  → Only ~8 of 26 are <50 yr (still fail C71 criterion)\n\n"
            "Root cause: autoencoder flags female multi-cancer patients\n"
            "as anomalous; C50 (breast) is the most common first-primary\n"
            "in women; any additional cancer bumps match_score to 0.5\n"
            "under the 2-site effective LFS definition.\n\n"
            "Fix for future screens:\n"
            "  • Require match_score ≥ 0.67 (≥2 of 3 original sites)\n"
            "  • Or require C71 explicitly as a mandatory criterion\n"
            "  • Or restrict to age_first < 46 (classic LFS)"
        )
        ax_r2.text(0.05, 0.95, lfs_note, transform=ax_r2.transAxes, fontsize=9,
                   va="top",
                   bbox=dict(facecolor="#fff0f0", edgecolor="#d62728", pad=6))
        ax_r2.set_title("LFS candidate failure analysis", fontsize=9)
        fig.suptitle("Li-Fraumeni syndrome — why all 26 candidates fail phenotypic refinement",
                     fontsize=11, color=NAVY)
        footer(ax_r2, 5)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 6: Survival + inter-cancer gap ────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        img(axes[0], R21/"fig_survival_curves.png",
            "Kaplan–Meier survival — Cowden vs LFS vs multi-cancer registry")
        img(axes[1], R21/"fig_inter_cancer_gap.png",
            "Inter-cancer gap (months) — Cowden match=1.0 vs LFS candidates")
        fig.suptitle("Survival and inter-cancer timing in syndrome candidates",
                     fontsize=11, color=NAVY)
        footer(axes[1], 6)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 7: Registry coverage gaps ────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 9))
        ax.axis("off")

        coverage_data = [
            ["Site","Name","In registry?","Impact"],
            ["C49","Soft tissue sarcoma","No","LFS: sarcoma not detectable"],
            ["C73","Thyroid","No","Cowden: thyroid arm missing"],
            ["C74","Adrenal cortex","No","LFS: adrenal tumour not detectable"],
            ["C64","Kidney","No","VHL: renal component missing"],
            ["C37","Thymus","No","MEN1: thymic neuroendocrine missing"],
            ["C71","Brain/CNS","Yes (n=718)","LFS detectable — but 0 candidates have it"],
            ["C50","Breast","Yes (n=7,432)","Cowden/LFS/BRCA: breast well captured"],
            ["C54","Endometrial","Yes (n=2,891)","Cowden/Lynch: endometrial well captured"],
            ["C18","Colorectal","Yes (n=8,244)","Lynch/FAP: colon well captured"],
        ]
        tbl3 = ax.table(
            cellText=coverage_data[1:],
            colLabels=coverage_data[0],
            cellLoc="center", loc="center",
            bbox=[0.0, 0.35, 1.0, 0.60]
        )
        tbl3.auto_set_font_size(False); tbl3.set_fontsize(9)
        for (r,c), cell in tbl3.get_celld().items():
            if r == 0: cell.set_facecolor(NAVY); cell.set_text_props(color="white")
            elif r <= 5: cell.set_facecolor("#fff3cd")  # missing sites
            else: cell.set_facecolor("#e8f5e9")          # present sites

        ax.text(0.5, 0.97, "Registry Site Coverage — Impact on Syndrome Detection",
                transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold", va="top")
        ax.text(0.05, 0.32,
                "Recommendation: Link cancer registry to pathology database\n"
                "to capture soft tissue (C49), thyroid (C73), and adrenal (C74) tumours.\n"
                "This would enable complete LFS and Cowden screening in ≥3-site patients.",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="#e3f2fd", edgecolor="#2e7fbf", pad=5))
        footer(ax, 7)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 8: Conclusions + next steps ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0,0.88),1,0.12,
                     transform=ax.transAxes, color=NAVY, zorder=0))
        ax.text(0.5, 0.935, "Conclusions and Recommendations",
                transform=ax.transAxes, ha="center", fontsize=14,
                color="white", fontweight="bold")
        ax.text(0.5, 0.895, "Hereditary Syndrome Candidate Validation — Taiwan Cancer Registry",
                transform=ax.transAxes, ha="center", fontsize=10, color="white")

        conclusions = (
            "Principal Conclusions\n\n"
            "1.  Registry-based autoencoder syndrome screening is feasible\n"
            "    but stringent phenotypic refinement is essential.\n"
            "    128 raw flagged candidates → 3 high-confidence after refinement.\n\n"
            "2.  Cowden syndrome (PTEN hamartoma) is the only syndrome\n"
            "    with registry-detectable high-confidence candidates.\n"
            "    Three patients (ages 46, 49, 57 yr) have both C50 (breast)\n"
            "    and C54 (endometrial) — the defining Cowden dyad.\n"
            "    These patients warrant PTEN germline testing referral.\n\n"
            "3.  Li-Fraumeni syndrome cannot be reliably detected in this\n"
            "    registry. All 26 flagged candidates have C50 + unrelated\n"
            "    cancer without C71 (brain) — the hallmark LFS tumour.\n"
            "    The 2-site effective definition (C50/C71) makes any\n"
            "    match_score=0.5 call clinically non-specific.\n\n"
            "4.  Three registry sites are critical missing data:\n"
            "    C49 (soft tissue sarcoma) — core LFS tumour type\n"
            "    C73 (thyroid) — third Cowden site\n"
            "    C74 (adrenal) — second LFS non-breast site\n\n"
            "Recommendations\n\n"
            "  SHORT TERM (current registry)\n"
            "    • Refer 3 Cowden high-confidence patients for PTEN testing\n"
            "    • Retire 'Li-Fraumeni' label from automated reports until\n"
            "      a mandatory C71 criterion is added\n"
            "    • Screen C50+C54 patients system-wide (not only top-1% anomalous)\n"
            "      — prevalence may be higher than autoencoder capture rate\n\n"
            "  MEDIUM TERM (registry expansion)\n"
            "    • Add C49, C73, C74 site codes to registry collection\n"
            "    • Link registry to pathology database for histological subtype\n"
            "      (e.g., dedifferentiated liposarcoma vs STS subtype)\n"
            "    • Add age-at-menarche, parity, BMI for Cowden risk scoring\n\n"
            "  LONG TERM (clinical validation)\n"
            "    • Prospective PTEN testing in all new C50+C54 patients\n"
            "    • Bi-directional cascade testing if germline mutation confirmed\n"
            "    • Compare Taiwan PTEN mutation spectrum to COSMIC/ClinVar"
        )
        ax.text(0.07, 0.82, conclusions, transform=ax.transAxes, fontsize=9,
                va="top", ha="left", fontfamily="monospace",
                bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", pad=8))
        footer(ax, 8)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

    size_kb = OUT.stat().st_size / 1024
    print(f"  Syndrome_Validation_Draft.pdf written — {size_kb:.0f} KB")
    print(f"  Path: {OUT}")


if __name__ == "__main__":
    main()
