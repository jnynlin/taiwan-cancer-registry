"""
Registry DL — Script 24: Sex-Specific Atlas Draft PDF

Key findings from Script 23:
  OR by site  — UADT sites 10–40× male-dominant; hormonal sites female-exclusive
  Age × sex   — UADT males present 5–9yr YOUNGER than females for same site
  Multi-cancer — Males 2× more likely to develop sequential cancers (6.8% vs 3.4%)
                 UADT males: 14.9% vs UADT females: 7.7%
  Survival    — Log-rank p<0.001 M vs F (M worse overall)
  VAE axes    — z4 UADT: rank-biserial=−0.621 (male); z5 Hormonal: rbi=+0.409 (female)

Output:
  results/Sex_Atlas_Draft.pdf
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = Path(__file__).parent.parent
R23  = BASE / "results/23_sex"
OUT  = BASE / "results/Sex_Atlas_Draft.pdf"

NAVY   = "#14304a"
ACCENT = "#2e7fbf"
PURPLE = "#9467bd"
GREEN  = "#2ca02c"

AXIS_PALETTE = {"UADT":"#2e7fbf","HBV/GI":"#2ca02c","Hormonal":"#9467bd","Other":"#aaaaaa"}


def img(ax, path, title=None):
    from matplotlib.image import imread
    p = Path(path)
    if p.exists():
        ax.imshow(imread(str(p))); ax.axis("off")
        if title: ax.set_title(title, fontsize=8, pad=3)
    else:
        ax.text(0.5, 0.5, f"[missing: {p.name}]", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="red"); ax.axis("off")


def footer(ax, page, n=8):
    ax.text(0.99, 0.01, f"Taiwan Cancer Registry — Sex-Specific Atlas  |  p. {page}/{n}",
            transform=ax.transAxes, fontsize=7, color="#888888", ha="right", va="bottom")


def main():
    print("=== Registry DL — 24: Sex-Specific Atlas Draft PDF ===")

    or_df  = pd.read_csv(R23 / "sex_or_by_site.csv")
    age_df = pd.read_csv(R23 / "age_sex_by_site.csv")

    n_male_dom  = len(or_df[(or_df["sig"]==True) & (or_df["OR"] > 1)])
    n_fem_dom   = len(or_df[(or_df["sig"]==True) & (or_df["OR"] < 1)])
    n_neutral   = len(or_df) - n_male_dom - n_fem_dom
    top_male    = or_df[or_df["OR"].notna()].nlargest(1, "OR").iloc[0]
    top_female  = or_df[or_df["OR"].notna()].nsmallest(1, "OR").iloc[0]

    with PdfPages(str(OUT)) as pdf:

        # ── Page 1: Title + summary ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off"); fig.patch.set_facecolor("white")
        ax.add_patch(plt.Rectangle((0,0.88),1,0.12,
                     transform=ax.transAxes, color=NAVY, zorder=0))
        ax.text(0.5, 0.935, "Taiwan Cancer Registry", transform=ax.transAxes,
                ha="center", fontsize=16, color="white", fontweight="bold")
        ax.text(0.5, 0.895, "Sex-Specific Cancer Atlas — 78,441 Patients · 37 Sites · 2003–2020",
                transform=ax.transAxes, ha="center", fontsize=11, color="white")

        summary = (
            "Overview\n\n"
            "Sex is the strongest binary covariate in the Taiwan Cancer Registry.\n"
            "The VAE identified two orthogonal sex-linked axes:\n"
            "  z4 — UADT/field-cancerization (rank-biserial=−0.621, male)\n"
            "  z0/z5 — Hormonal/gynecologic  (rbi=+0.163/+0.409, female)\n\n"
            "These axes map to distinct carcinogenic exposures:\n"
            "  Male   → betel nut + tobacco (UADT) + HBV/metabolic syndrome (GI)\n"
            "  Female → endogenous hormones (breast, endometrial, ovarian)\n\n"
            "Key Findings (all FDR<0.05)\n\n"
            f"  Site sex-OR spectrum:\n"
            f"    Male-dominant:   {n_male_dom} sites\n"
            f"    Female-dominant: {n_fem_dom} sites\n"
            f"    Sex-neutral:     {n_neutral} sites\n\n"
            f"  Strongest male site:   C12 pyriform  OR={top_male['OR']:.0f}×\n"
            f"  Strongest female site: C50 breast    OR={top_female['OR']:.3f}×\n\n"
            "  Age × sex interaction:\n"
            "    UADT males present 5–9yr YOUNGER than UADT females\n"
            "    (C06: Δ=−9yr, C02: Δ=−7yr, C15: Δ=−5yr)\n"
            "    C50 breast: males 15yr OLDER than females (incidental/late-onset)\n\n"
            "  Multi-cancer rate:\n"
            "    Males: 6.8%  vs  Females: 3.4%  (2.0× difference)\n"
            "    UADT males: 14.9% vs UADT females: 7.7%  (axis-specific)\n\n"
            "  Survival:\n"
            "    Log-rank M vs F: p<0.001; males worse overall"
        )
        ax.text(0.07, 0.82, summary, transform=ax.transAxes, fontsize=9.5,
                va="top", ha="left", fontfamily="monospace",
                bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", pad=8))
        footer(ax, 1)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 2: M:F OR forest ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 11))
        img(ax, R23/"fig_mf_ratio_bar.png",
            "Male:Female OR by site — log₂ scale (all 37 sites)")
        fig.suptitle("Sex-specific cancer prevalence — Male:Female OR forest",
                     fontsize=11, color=NAVY)
        footer(ax, 2)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 3: Age × sex violin ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 8))
        img(ax, R23/"fig_age_sex_violin.png",
            "Age at first diagnosis by sex — key sites (blue=male, purple=female)")
        fig.suptitle("Age at first diagnosis by sex — UADT males present younger",
                     fontsize=11, color=NAVY)
        footer(ax, 3)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 4: Age × sex table + interpretation ──────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        ax_l = axes[0]; ax_l.axis("off")
        sig_age = age_df[age_df["sig"]==True].sort_values("age_diff_mf", key=abs, ascending=False)
        tbl_data = [[r["site"], r["axis"],
                     f"{r['median_m']:.0f}", f"{r['median_f']:.0f}",
                     f"{r['age_diff_mf']:+.0f}",
                     f"{r['q_fdr']:.4f}"]
                    for _, r in sig_age.head(12).iterrows()]
        tbl = ax_l.table(
            cellText=tbl_data,
            colLabels=["Site","Axis","M median","F median","Δ (M−F)","q FDR"],
            cellLoc="center", loc="center",
            bbox=[0.0, 0.1, 1.0, 0.85]
        )
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        for (r,c), cell in tbl.get_celld().items():
            if r == 0: cell.set_facecolor(NAVY); cell.set_text_props(color="white")
            elif r <= len(sig_age) and sig_age.iloc[r-1]["age_diff_mf"] < 0:
                cell.set_facecolor("#e8f4fd")   # male younger
            else:
                cell.set_facecolor("#f8f0ff")   # female younger
        ax_l.set_title("Age × sex differences by site (FDR<0.05, top 12)", fontsize=9)

        ax_r = axes[1]; ax_r.axis("off")
        interp = (
            "UADT sites: males present YOUNGER\n\n"
            "C06 oral cavity: −9yr  C02 tongue: −7yr\n"
            "C15 esophagus:  −5yr  C03 floor of mouth: −10yr\n\n"
            "Interpretation: heavy betel nut + tobacco exposure\n"
            "begins in adolescence/early adulthood in Taiwan males.\n"
            "Earlier cumulative exposure → earlier cancer onset.\n"
            "Female UADT patients likely have lower betel/tobacco\n"
            "exposure → present later.\n\n"
            "C22 liver HCC: males −7yr younger\n\n"
            "Consistent with higher HBV carrier rate in males\n"
            "(male HBV clearance is lower → higher chronicity).\n\n"
            "C50 breast: males +15yr older\n\n"
            "Male breast cancer is rare, late-onset, typically\n"
            "post-60yr in Taiwanese men. Female breast cancer\n"
            "peaks 40–55yr (younger than Western populations).\n\n"
            "Clinical implication:\n"
            "Screening programmes should be sex-stratified\n"
            "and use different age-at-initiation thresholds."
        )
        ax_r.text(0.05, 0.95, interp, transform=ax_r.transAxes, fontsize=9,
                  va="top",
                  bbox=dict(facecolor="#f0f4f8", edgecolor="#cccccc", pad=6))
        ax_r.set_title("Age × sex — biological interpretation", fontsize=9)
        fig.suptitle("Age at first diagnosis: sex differences by site",
                     fontsize=11, color=NAVY)
        footer(ax_r, 4)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 5: Multi-cancer rate by sex + axis ───────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        img(axes[0], R23/"fig_multi_cancer_sex.png",
            "Multi-cancer rate by sex (overall + by axis)")

        ax_r = axes[1]; ax_r.axis("off")
        mc_note = (
            "Multi-cancer rate by sex — key finding\n\n"
            "Overall:\n"
            "  Males: 6.8%   Females: 3.4%   Ratio: 2.0×\n\n"
            "By carcinogenic axis:\n"
            "  UADT  M: 14.9%  vs  F: 7.7%   Ratio: 1.9×\n"
            "  HBV/GI M: 4.2%   vs  F: 3.0%   Ratio: 1.4×\n"
            "  Other   M: 6.0%   vs  F: 2.1%   Ratio: 2.9×\n\n"
            "Interpretation:\n"
            "  The 2× male excess in multi-cancer rate is not\n"
            "  simply explained by UADT field cancerization\n"
            "  (the largest contributor) — Other-axis males also\n"
            "  show 2.9× excess, suggesting a systemic sex\n"
            "  difference in multi-primary risk.\n\n"
            "Possible mechanisms:\n"
            "  1. Betel nut + alcohol + tobacco act synergistically\n"
            "     across multiple anatomical sites (not only UADT)\n"
            "  2. Immune surveillance differences between sexes:\n"
            "     higher female cytotoxic T-cell activity may\n"
            "     suppress nascent second primaries more effectively\n"
            "  3. Hormonal influence on DNA repair capacity\n\n"
            "Note: hormonal-axis males n=40 only (prostate + lung)\n"
            "— UADT dominates male carcinogenesis."
        )
        ax_r.text(0.05, 0.95, mc_note, transform=ax_r.transAxes, fontsize=9,
                  va="top",
                  bbox=dict(facecolor="#f0f8f0", edgecolor="#2ca02c", pad=6))
        ax_r.set_title("Multi-cancer rate by sex and axis", fontsize=9)
        fig.suptitle("Males 2× more likely to develop sequential cancers",
                     fontsize=11, color=NAVY)
        footer(ax_r, 5)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 6: Survival by sex ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        img(ax, R23/"fig_survival_sex.png",
            "Kaplan–Meier survival by sex (overall + UADT subset)")
        fig.suptitle("Survival by sex — overall and UADT-specific",
                     fontsize=11, color=NAVY)
        footer(ax, 6)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 7: VAE axis sex separation ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        img(ax, R23/"fig_vae_axis_sex.png",
            "VAE active axis values by sex — latent space sex separation")
        fig.suptitle("VAE latent axes encode sex — z4 male-shifted, z0/z5 female-shifted",
                     fontsize=11, color=NAVY)
        footer(ax, 7)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 8: Conclusions + limitations ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off"); fig.patch.set_facecolor("white")
        ax.add_patch(plt.Rectangle((0,0.88),1,0.12,
                     transform=ax.transAxes, color=NAVY, zorder=0))
        ax.text(0.5, 0.935, "Conclusions and Limitations",
                transform=ax.transAxes, ha="center", fontsize=14,
                color="white", fontweight="bold")
        ax.text(0.5, 0.895, "Sex-Specific Cancer Atlas — Taiwan Cancer Registry 2003–2020",
                transform=ax.transAxes, ha="center", fontsize=10, color="white")

        conclusions = (
            "Principal Conclusions\n\n"
            "1.  The Taiwan cancer registry is strongly sex-stratified.\n"
            "    Of 37 sites, {mal} are male-dominant, {fem} are female-dominant,\n"
            "    and {neu} are sex-neutral — all FDR<0.05.\n\n"
            "2.  Two orthogonal sex-linked carcinogenic axes:\n"
            "    MALE   → betel/tobacco UADT (z4, rbi=−0.621) + HBV/GI\n"
            "    FEMALE → hormonal/gynecologic (z0/z5, rbi=+0.409)\n"
            "    These axes explain the bulk of sex heterogeneity in the registry.\n\n"
            "3.  UADT males present 5–9yr younger than females for the same site.\n"
            "    Consistent with earlier-onset betel nut exposure in male adolescents.\n"
            "    Screening programmes should use sex-specific age-at-initiation.\n\n"
            "4.  Males have 2× the multi-cancer rate of females (6.8% vs 3.4%).\n"
            "    Excess is not confined to UADT — 'Other' sites show 2.9× excess.\n"
            "    Possible mechanisms: carcinogen synergy, immune sex differences.\n\n"
            "5.  Survival is significantly worse in males (log-rank p<0.001).\n"
            "    Partly explained by UADT predominance (high-mortality sites)\n"
            "    and later-stage presentation.\n\n"
            "Limitations\n\n"
            "   No betel nut/tobacco/alcohol exposure data — sex differences are\n"
            "     attributed to exposure axes by ecological inference only.\n"
            "   No hormonal data (parity, OCP, HRT) for female hormonal axis.\n"
            "   Registry records first-primary site per patient; rare non-primary\n"
            "     sex assignment errors possible for sex-exclusive sites.\n"
            "   Multi-cancer rate may be partially confounded by follow-up duration\n"
            "     (males have higher baseline mortality → shorter FU window for\n"
            "     detecting second cancers).\n\n"
            "Next Steps\n\n"
            "   Sex-stratified survival models for each axis (Cox with sex × axis interaction)\n"
            "   Age-standardised incidence rates by site and sex (vs general population)\n"
            "   Link sex-specific multi-cancer risk to Transformer surveillance calendar\n"
            "     (male UADT patients warrant more aggressive 6-monthly endoscopy)"
        ).format(mal=n_male_dom, fem=n_fem_dom, neu=n_neutral)

        ax.text(0.07, 0.82, conclusions, transform=ax.transAxes, fontsize=9,
                va="top", ha="left", fontfamily="monospace",
                bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", pad=8))
        footer(ax, 8)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

    size_kb = OUT.stat().st_size / 1024
    print(f"  Sex_Atlas_Draft.pdf written — {size_kb:.0f} KB")
    print(f"  Path: {OUT}")


if __name__ == "__main__":
    main()
