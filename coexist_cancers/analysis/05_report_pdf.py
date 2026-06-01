"""
Assemble a multi-page PDF report for the Co-existing Cancer Pattern Discovery project.
Embeds the figures produced by steps 01–04 with text framing.
Output: results/Coexist_Cancers_Report.pdf
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

BASE = Path(__file__).parent.parent
R    = BASE / "results"
OUT  = R / "Coexist_Cancers_Report.pdf"

TITLE_BG = "#14304a"
ACCENT   = "#2e7fbf"
GRAY     = "#555555"
BG_LIGHT = "#eef4fb"
A4 = (8.27, 11.69)
TOTAL = 6


def header(fig, title, subtitle=""):
    fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=15,
             fontweight="bold", color=TITLE_BG)
    if subtitle:
        fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=9.5,
                 color=GRAY, style="italic")

def footer(fig, n):
    fig.text(0.5, 0.012,
             f"Taiwan Cancer Registry 2006–2020  ·  Co-existing Cancer Patterns  ·  Page {n}/{TOTAL}",
             ha="center", va="bottom", fontsize=7, color=GRAY)

def img(fig, path, rect, caption=""):
    p = R / path
    if not p.exists():
        return
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(str(p)))
    ax.axis("off")
    if caption:
        ax.set_title(caption, fontsize=8.5, color=GRAY, pad=3)

def statbox(ax, value, label, sub=""):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.03,0.05),0.94,0.9, boxstyle="round,pad=0.02",
                 linewidth=1.2, edgecolor=ACCENT, facecolor=BG_LIGHT))
    ax.text(0.5,0.62,value,ha="center",va="center",fontsize=18,fontweight="bold",color=ACCENT)
    ax.text(0.5,0.30,label,ha="center",va="center",fontsize=8.5,fontweight="bold",color=TITLE_BG)
    if sub:
        ax.text(0.5,0.13,sub,ha="center",va="center",fontsize=6.8,color=GRAY)


with PdfPages(OUT) as pdf:

    # ── PAGE 1: Title + overview ──────────────────────────────────────────────
    fig = plt.figure(figsize=A4); footer(fig, 1)
    axt = fig.add_axes([0.0, 0.80, 1.0, 0.17]); axt.set_facecolor(TITLE_BG)
    axt.set_xticks([]); axt.set_yticks([])
    for sp in axt.spines.values(): sp.set_visible(False)
    axt.text(0.5, 0.62,
             "Unsupervised Discovery of Co-existing Cancer Patterns\nin the Taiwan Cancer Registry",
             ha="center", va="center", fontsize=15, fontweight="bold", color="white",
             linespacing=1.5, transform=axt.transAxes)
    axt.text(0.5, 0.20,
             "Association Rule Mining · NMF Cancer Programs · Deep Autoencoder Clustering",
             ha="center", va="center", fontsize=9.5, color="#aaccee", transform=axt.transAxes)

    stats = [("78,578","Patients","2006–2020"),
             ("4,068","Multi-primary","≥2 cancer types (5.2%)"),
             ("46","Cancer sites","ICD-O C-codes"),
             ("3","Co-occurrence\ndomains","convergent")]
    for i,(v,l,s) in enumerate(stats):
        statbox(fig.add_axes([0.05+i*0.235, 0.66, 0.21, 0.11]), v, l, s)

    ax = fig.add_axes([0.07, 0.08, 0.86, 0.54]); ax.axis("off")
    body = [
        ("OVERVIEW", True, 11),
        ("", False, 4),
        ("We applied three complementary unsupervised machine-learning methods to the full "
         "Taiwan Cancer Registry (78,578 patients, all cancer types) to discover which cancers "
         "co-occur within the same patient more often than expected by chance. Of the cohort, "
         "4,068 patients (5.2%) developed two or more distinct primary cancers; 536 developed "
         "three or more.", False, 9),
        ("", False, 3),
        ("Methods", True, 9.5),
        ("(1) Pairwise association rule mining computed lift and odds ratios for every cancer "
         "pair, stratified by sex and age of onset. (2) Non-negative matrix factorization (NMF) "
         "decomposed the multi-primary patient × cancer matrix into latent 'cancer programs'. "
         "(3) A binary-cross-entropy autoencoder learned a 12-dimensional latent representation, "
         "followed by UMAP visualization and k-means clustering.", False, 9),
        ("", False, 3),
        ("Principal finding", True, 9.5),
        ("All three independent methods converged on the same three co-occurrence domains: "
         "(I) an aerodigestive squamous-cell-carcinoma field — esophagus, oral cavity, "
         "hypopharynx, larynx — that is 95–97% male, has the youngest onset (median 55 yr), "
         "the worst survival (68.8% mortality), and the highest rate of a third primary cancer "
         "(13%); (II) a female reproductive-tract and breast cluster; and (III) an age-related "
         "visceral GI / urological cluster. The aerodigestive domain is the clinical fingerprint "
         "of Taiwan's betel-nut / alcohol / tobacco field cancerization.", False, 9),
        ("", False, 3),
        ("Significance", True, 9.5),
        ("Convergence of association rules, NMF, and deep clustering on the same partition "
         "provides strong evidence that these co-occurrence domains reflect real shared "
         "carcinogenic mechanisms rather than statistical artifacts — with direct implications "
         "for second-cancer surveillance in high-risk patients.", False, 9),
    ]
    import textwrap as tw
    y = 1.0
    for text, bold, size in body:
        if not text:
            y -= size/520; continue
        wrapped = tw.fill(text, 96) if (not bold and len(text)>80) else text
        ax.text(0, y, wrapped, transform=ax.transAxes, fontsize=size,
                fontweight="bold" if bold else "normal",
                color=TITLE_BG if bold else "#111111", va="top", linespacing=1.5)
        y -= (size*(wrapped.count("\n")+1)*1.6)/520
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 2: Cohort & co-occurrence landscape ──────────────────────────────
    fig = plt.figure(figsize=A4)
    header(fig, "Figure 1. Cohort & Co-occurrence Landscape",
           "Cancer prevalence and pairwise co-occurrence among multi-primary patients")
    footer(fig, 2)
    img(fig, "01_matrix/cancer_prevalence.png",  [0.04,0.60,0.92,0.30], "Top 30 cancer sites by patient prevalence")
    img(fig, "01_matrix/cooccurrence_heatmap.png",[0.04,0.10,0.92,0.42], "Cancers per patient (left) · co-occurrence counts, top 25 sites (right)")
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 3: Association rules ──────────────────────────────────────────────
    fig = plt.figure(figsize=A4)
    header(fig, "Figure 2. Association Rule Mining",
           "Cancer pairs co-occurring more than expected (lift > 1)")
    footer(fig, 3)
    img(fig, "02_associations/top_associations_lift.png", [0.03,0.58,0.94,0.33], "Top cancer pairs by lift (bar) and support–lift bubble plot")
    img(fig, "02_associations/lift_heatmap.png",          [0.10,0.08,0.80,0.46], "Lift matrix — top sites (red = co-occur more than chance)")
    fig.text(0.5, 0.045,
             "Strongest pair: Hypopharynx ↔ Esophagus (n=156, lift=5.50, OR=6.77). "
             "29 male-specific vs 5 female-specific high-lift pairs — co-occurrence is "
             "overwhelmingly a male aerodigestive phenomenon.",
             ha="center", fontsize=7.5, color=GRAY, style="italic", wrap=True)
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 4: NMF programs ──────────────────────────────────────────────────
    fig = plt.figure(figsize=A4)
    header(fig, "Figure 3. NMF Cancer Co-occurrence Programs",
           "Latent decomposition of the multi-primary cohort (k=7)")
    footer(fig, 4)
    img(fig, "03_nmf/nmf_programs_multiprimary.png", [0.03,0.30,0.94,0.60], "Seven cancer programs — top sites per component")
    ax = fig.add_axes([0.06,0.06,0.88,0.20]); ax.axis("off")
    rows = [["Program","Theme","n","≥3 cancers","Age","Male%"],
            ["P2","Aerodigestive SCC (esoph+pharynx)","615","17%","53","97%"],
            ["P5","Oral cavity field cancerization","758","19%","52","95%"],
            ["P6","Female genital / breast","730","3%","56","5%"],
            ["P4","Liver / GI","766","7%","64","69%"],
            ["P7","Urological (prostate+bladder)","406","5%","72","90%"]]
    t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center",
                 bbox=[0,0,1,1], colWidths=[0.10,0.42,0.10,0.16,0.10,0.12])
    t.auto_set_font_size(False); t.set_fontsize(7.5)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r==0: cell.set_facecolor(ACCENT); cell.set_text_props(color="white",fontweight="bold")
        elif rows[r][0] in ("P2","P5"): cell.set_facecolor("#ffe9d6")
        elif r%2==0: cell.set_facecolor(BG_LIGHT)
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 5: Transitions + deep clustering ─────────────────────────────────
    fig = plt.figure(figsize=A4)
    header(fig, "Figure 4. Transitions & Deep-Learning Clusters",
           "1st→2nd primary flow and autoencoder/UMAP patient subtypes")
    footer(fig, 5)
    img(fig, "02_associations/cancer_transition_matrix.png", [0.03,0.62,0.94,0.30], "1st → 2nd primary cancer transitions")
    img(fig, "04_clustering/umap_nmf_programs.png",          [0.02,0.32,0.48,0.28], "UMAP — NMF program overlay")
    img(fig, "04_clustering/cluster_cancer_heatmap_k3.png",  [0.50,0.34,0.49,0.24], "Cancer prevalence by AE cluster")
    ax = fig.add_axes([0.06,0.06,0.88,0.21]); ax.axis("off")
    rows = [["Cluster","Theme","n","Median age","Male%","Mortality","≥3 cancers"],
            ["C3","Aerodigestive SCC field","1,071","55","96%","68.8%","13%"],
            ["C1","GI / visceral mixed","1,827","61","76%","58.8%","11%"],
            ["C2","Female-enriched / breast","1,170","61","30%","49.1%","4%"]]
    t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center",
                 bbox=[0,0,1,0.9], colWidths=[0.10,0.30,0.10,0.14,0.10,0.14,0.12])
    t.auto_set_font_size(False); t.set_fontsize(7.5)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r==0: cell.set_facecolor(ACCENT); cell.set_text_props(color="white",fontweight="bold")
        elif rows[r][0]=="C3": cell.set_facecolor("#ffe9d6")
        elif r%2==0: cell.set_facecolor(BG_LIGHT)
    ax.text(0.5, 0.97, "k-means clusters on autoencoder latent space (k=3)",
            ha="center", fontsize=8.5, fontweight="bold", color=TITLE_BG, transform=ax.transAxes)
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 6: Conclusions ───────────────────────────────────────────────────
    fig = plt.figure(figsize=A4); footer(fig, 6)
    ax = fig.add_axes([0.07,0.06,0.86,0.90]); ax.axis("off")
    secs = [
        ("CONCLUSIONS", True, 12),
        ("", False, 4),
        ("Three convergent co-occurrence domains", True, 10),
        ("1.  Aerodigestive SCC field cancerization — esophagus + oral cavity + hypopharynx + "
         "larynx. Identified independently as 29 male association pairs, NMF programs P2+P5, "
         "and autoencoder cluster C3. Male 95–97%, youngest onset (52–55 yr), worst survival "
         "(68.8% mortality), highest 3rd-cancer risk (13–19%). Driven by betel nut, tobacco, "
         "and alcohol — a Taiwan-specific lifestyle signature.", False, 9),
        ("", False, 2),
        ("2.  Female reproductive tract + breast — breast ↔ cervix ↔ corpus uteri ↔ ovary "
         "(NMF P6, cluster C2). Best survival of the three domains; reflects shared hormonal "
         "and genetic susceptibility.", False, 9),
        ("", False, 2),
        ("3.  Age-related visceral GI / urological — colorectal, liver, lung, prostate, bladder "
         "(NMF P1/P3/P4/P7, cluster C1). Older patients, both sexes, surveillance-driven "
         "second-cancer detection.", False, 9),
        ("", False, 4),
        ("Clinical implications", True, 10),
        ("Patients presenting with any aerodigestive SCC warrant intensified pan-aerodigestive "
         "surveillance (panendoscopy of oral cavity, pharynx, larynx, and esophagus), given the "
         "13–19% rate of a third primary. The convergence across three unsupervised methods "
         "strengthens confidence that these are mechanistically real domains, not statistical "
         "artifacts.", False, 9),
        ("", False, 4),
        ("Limitations", True, 10),
        ("Same-site recurrence vs true second primary cannot be separated from registry codes; "
         "metastasis may contaminate pairs such as Colon→Liver. Cancer sequence numbering may "
         "undercount later primaries. The all-patient UMAP uses a 12k subsample for embedding "
         "(visualization only; clustering uses the full latent space).", False, 9),
        ("", False, 4),
        ("Reproduce", True, 10),
        ("bash coexist_cancers/analysis/run_all.sh   →   01 matrix · 02 association rules · "
         "03 NMF · 04 autoencoder clustering · 05 this report", False, 8.5),
    ]
    y = 1.0
    for text, bold, size in secs:
        if not text:
            y -= size/560; continue
        wrapped = tw.fill(text, 100) if (not bold and len(text)>80) else text
        ax.text(0, y, wrapped, transform=ax.transAxes, fontsize=size,
                fontweight="bold" if bold else "normal",
                color=TITLE_BG if bold else "#111111", va="top", linespacing=1.5)
        y -= (size*(wrapped.count("\n")+1)*1.55)/560
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

print(f"PDF saved: {OUT}  ({OUT.stat().st_size//1024} KB)")
