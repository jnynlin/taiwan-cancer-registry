"""
Near-submission manuscript for the Co-existing Cancer Pattern Discovery study.
Target journal: Cancer Epidemiology (Elsevier) — structured abstract ≤300 words.
Output: results/Coexist_Cancers_Manuscript.pdf
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import textwrap as tw
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

BASE = Path(__file__).parent.parent
R    = BASE / "results"
OUT  = R / "Coexist_Cancers_Manuscript.pdf"

NAVY = "#14304a"; ACCENT = "#2e7fbf"; GRAY = "#555555"; BG = "#eef4fb"
A4 = (8.27, 11.69)
TOTAL = 9


def footer(fig, n):
    fig.text(0.5, 0.012,
             f"Co-existing Cancer Patterns · Taiwan Cancer Registry 2006–2020 · Page {n}/{TOTAL}",
             ha="center", va="bottom", fontsize=7, color=GRAY)

def img(fig, path, rect, cap=""):
    p = R / path
    if not p.exists(): return
    ax = fig.add_axes(rect); ax.imshow(mpimg.imread(str(p))); ax.axis("off")
    if cap: ax.set_title(cap, fontsize=8, color=GRAY, pad=2)

def flow(ax, blocks, width=100, top=1.0, scale=560):
    """Render (text, bold, size) blocks with wrapping; returns final y."""
    y = top
    for text, bold, size in blocks:
        if not text:
            y -= size/scale; continue
        wrapped = tw.fill(text, width) if (not bold and len(text) > 80) else text
        ax.text(0, y, wrapped, transform=ax.transAxes, fontsize=size,
                fontweight="bold" if bold else "normal",
                color=NAVY if bold else "#111111", va="top", linespacing=1.5)
        y -= (size*(wrapped.count("\n")+1)*1.55)/scale
    return y


with PdfPages(OUT) as pdf:

    # ── PAGE 1: Title + structured abstract ───────────────────────────────────
    fig = plt.figure(figsize=A4); footer(fig, 1)
    axt = fig.add_axes([0.0, 0.83, 1.0, 0.14]); axt.set_facecolor(NAVY)
    axt.set_xticks([]); axt.set_yticks([])
    for sp in axt.spines.values(): sp.set_visible(False)
    axt.text(0.5, 0.60,
             "Convergent Unsupervised Machine Learning Reveals an Aerodigestive\n"
             "Field-Cancerization Domain Driving Multiple Primary Cancers in Taiwan",
             ha="center", va="center", fontsize=12.5, fontweight="bold", color="white",
             linespacing=1.5, transform=axt.transAxes)
    axt.text(0.5, 0.16, "[Author Names, Degrees]   ·   [Institution, City, Taiwan]   ·   [Corresponding: email]",
             ha="center", va="center", fontsize=8, color="#aaccee", transform=axt.transAxes)

    ax = fig.add_axes([0.07, 0.06, 0.86, 0.74]); ax.axis("off")
    blocks = [
        ("ABSTRACT", True, 11), ("", False, 4),
        ("Background", True, 9.5),
        ("Multiple primary cancers (MPC) are increasingly common, yet the patterns by which "
         "specific cancers co-occur within the same patient — and the shared carcinogenic "
         "mechanisms they imply — remain poorly characterized in Asian populations where betel "
         "nut, tobacco, and alcohol exposures are prevalent. We applied three independent "
         "unsupervised machine-learning methods to a large cancer registry to discover "
         "co-existing cancer patterns.", False, 9),
        ("", False, 3),
        ("Methods", True, 9.5),
        ("We analyzed 78,578 patients (84,157 primary cancers, ICD-O site codes) from the Taiwan "
         "Cancer Registry (2006–2020). A multi-hot patient × cancer matrix was constructed. "
         "Pairwise association rules (lift, odds ratio) were computed and stratified by sex and "
         "age of onset. Non-negative matrix factorization (NMF) decomposed the multi-primary "
         "cohort into latent cancer programs. A binary-cross-entropy autoencoder learned a "
         "12-dimensional latent representation, followed by UMAP and k-means clustering. "
         "Concordance across the three methods was assessed.", False, 9),
        ("", False, 3),
        ("Results", True, 9.5),
        ("4,068 patients (5.2%) developed ≥2 distinct primary cancers; 536 (0.7%) developed ≥3. "
         "Across 288,424 person-years, 69 index→target pairs showed significantly elevated "
         "second-primary risk (FDR<0.05), the highest being esophagus→larynx (SIR 12.2) and "
         "esophagus→hypopharynx (SIR 12.0). All three unsupervised methods converged on three "
         "domains: (I) an aerodigestive squamous-cell-carcinoma field (esophagus, oral cavity, "
         "hypopharynx, larynx) — 96% male, youngest onset (median 55 y), highest mortality "
         "(68.8%), highest third-cancer rate (13%); (II) a female reproductive-tract/breast "
         "cluster (49.1% mortality); and (III) an age-related visceral/urological cluster. "
         "Trajectory mining distinguished two temporal modes: aerodigestive pairs were "
         "bidirectional and synchronous (60% within 6 months), whereas breast and colorectal "
         "second primaries were unidirectional and metachronous (10–24% synchronous).", False, 9),
        ("", False, 3),
        ("Conclusions", True, 9.5),
        ("Three orthogonal unsupervised methods independently identified the same aerodigestive "
         "field-cancerization domain as the dominant driver of MPC in Taiwan. Patients with any "
         "aerodigestive squamous-cell carcinoma warrant intensified pan-aerodigestive "
         "surveillance.", False, 9),
        ("", False, 3),
        ("Keywords", True, 9),
        ("Multiple primary cancers; Field cancerization; Esophageal cancer; Association rule "
         "mining; Non-negative matrix factorization; Deep learning; Cancer registry; Taiwan",
         False, 8.5),
    ]
    flow(ax, blocks, width=98, top=1.0, scale=560)
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 2: Introduction ──────────────────────────────────────────────────
    fig = plt.figure(figsize=A4); footer(fig, 2)
    ax = fig.add_axes([0.07, 0.05, 0.86, 0.92]); ax.axis("off")
    blocks = [
        ("1.  INTRODUCTION", True, 12), ("", False, 5),
        ("Improvements in cancer survival and the aging of populations have produced a growing "
         "cohort of patients who develop more than one primary malignancy in their lifetime. "
         "Multiple primary cancers (MPC) now account for an estimated 8–17% of new cancer "
         "diagnoses in registry studies, and their incidence is rising. MPC may arise from shared "
         "environmental carcinogen exposure, heritable predisposition, prior treatment effects, "
         "or intensified surveillance — but disentangling which cancers co-occur, and why, "
         "requires population-scale data and methods that can surface structure without "
         "prior hypotheses.", False, 9.5),
        ("", False, 4),
        ("The concept of field cancerization — whereby a carcinogen-exposed epithelial field "
         "accumulates multiple independent transformation events — is particularly relevant to "
         "the upper aerodigestive tract. In Taiwan, the combined habits of betel quid chewing, "
         "cigarette smoking, and alcohol consumption create a uniquely high-risk mucosal field "
         "spanning the oral cavity, pharynx, larynx, and esophagus. Squamous-cell carcinomas "
         "(SCC) arising at these sites share histology and risk factors, and second primaries "
         "within this field are well documented clinically. However, the global structure of "
         "cancer co-occurrence across all sites — and whether data-driven methods recover "
         "clinically meaningful domains — has not been systematically characterized in this "
         "population.", False, 9.5),
        ("", False, 4),
        ("Unsupervised machine learning offers a hypothesis-free lens on such structure. "
         "Association rule mining quantifies which cancer pairs co-occur more than expected by "
         "chance. Non-negative matrix factorization (NMF), widely used to extract mutational "
         "signatures, can analogously decompose patient–cancer co-occurrence into interpretable "
         "latent programs. Deep autoencoders learn compact nonlinear representations that, "
         "combined with clustering, can reveal patient subtypes invisible to pairwise methods. "
         "Crucially, when independent methods converge on the same structure, confidence that "
         "the structure is biologically real — rather than a methodological artifact — is "
         "substantially strengthened.", False, 9.5),
        ("", False, 4),
        ("In this study we applied all three approaches to 78,578 patients from the Taiwan Cancer "
         "Registry (2006–2020). We aimed to (1) quantify the landscape of cancer co-occurrence; "
         "(2) identify latent co-occurrence programs and patient subtypes; and (3) test whether "
         "association rules, NMF, and deep clustering converge on a common set of domains and "
         "their clinical correlates.", False, 9.5),
    ]
    flow(ax, blocks, width=100, top=1.0, scale=560)
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 3: Methods ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=A4); footer(fig, 3)
    ax = fig.add_axes([0.07, 0.05, 0.86, 0.92]); ax.axis("off")
    blocks = [
        ("2.  MATERIALS AND METHODS", True, 12), ("", False, 5),
        ("2.1  Data source and cohort", True, 9.5),
        ("De-identified records were obtained from the Taiwan Cancer Registry long-form dataset "
         "(2006–2020). Each record encodes an ICD-O topography code, morphology, diagnosis date, "
         "cancer sequence number, and vital status. Tumor sites were collapsed to three-character "
         "ICD-O C-codes; sites with ≥10 patients (46 sites) were retained. The analysis cohort "
         "comprised 78,578 patients contributing 84,157 primary cancers.", False, 9.5),
        ("", False, 3),
        ("2.2  Patient × cancer matrix", True, 9.5),
        ("A binary multi-hot matrix was constructed (rows = patients, columns = cancer sites; "
         "1 = ever diagnosed). Patients with ≥2 distinct sites were defined as multiple-primary "
         "(MPC). For methods requiring richer signal, the MPC subset (n=4,068) was used, "
         "restricted to sites with ≥3 MPC patients (39 sites).", False, 9.5),
        ("", False, 3),
        ("2.3  Association rule mining", True, 9.5),
        ("For every cancer pair we computed support, confidence, lift (observed/expected "
         "co-occurrence), and the odds ratio with a 2×2 contingency table. Pairs with lift > 1.5 "
         "were considered enriched. Analyses were repeated stratified by sex and by age of onset "
         "(<50 vs ≥60 years).", False, 9.5),
        ("", False, 3),
        ("2.4  Non-negative matrix factorization", True, 9.5),
        ("NMF (scikit-learn, nndsvda initialization, L1 ratio 0.05) decomposed the MPC matrix into "
         "k latent programs. k was selected by reconstruction error and cosine silhouette across "
         "k = 2–8; k = 7 was chosen. Each program is a non-negative weighting over cancer sites; "
         "patients were assigned to their dominant program for clinical profiling.", False, 9.5),
        ("", False, 3),
        ("2.5  Autoencoder and clustering", True, 9.5),
        ("A symmetric autoencoder (encoder 64-32-12; sigmoid decoder; binary-cross-entropy loss; "
         "Adam, cosine-annealed learning rate; 150 epochs) was trained on the MPC cohort and used "
         "to embed all patients into a 12-dimensional latent space. UMAP (cosine metric; "
         "subsampled to 12,000 for tractable embedding) provided 2-D visualization. k-means "
         "clustering used cosine silhouette to select k (k = 3).", False, 9.5),
        ("", False, 3),
        ("2.6  Second-primary SIR and trajectory mining", True, 9.5),
        ("Standardized incidence ratios (SIR) quantified second-primary risk: from each patient's "
         "first cancer, person-years at risk accrued from index diagnosis + 2-month washout to "
         "death, last contact, or 2020-12-31. Expected counts used sex- and age-at-index-stratified "
         "internal-reference background rates; SIR = observed/expected with Poisson exact 95% CI, "
         "FDR-corrected. Directional trajectories (A→B where B followed A) were tested by binomial "
         "sign test; second primaries within 6 months were classed synchronous, otherwise "
         "metachronous; three-step paths were enumerated.", False, 9.5),
        ("", False, 2),
        ("2.7  Convergence and statistics", True, 9.5),
        ("Concordance among methods was assessed by mapping NMF programs and k-means clusters onto "
         "the association-rule domains. Analyses used Python 3.11 (pandas, scikit-learn, PyTorch, "
         "umap-learn, statsmodels, networkx).", False, 9.5),
    ]
    flow(ax, blocks, width=100, top=1.0, scale=590)
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 4: Results I — landscape + association ───────────────────────────
    fig = plt.figure(figsize=A4); footer(fig, 4)
    ax = fig.add_axes([0.07, 0.74, 0.86, 0.22]); ax.axis("off")
    flow(ax, [
        ("3.  RESULTS", True, 12), ("", False, 4),
        ("3.1  Co-occurrence landscape and association rules", True, 9.5),
        ("Of 78,578 patients, 4,068 (5.2%) developed ≥2 distinct primary cancers and 536 (0.7%) "
         "developed ≥3 (maximum 7). Forty-five cancer pairs showed lift > 1.5. The strongest "
         "were confined to the upper aerodigestive tract: pyriform sinus–esophagus (lift 8.11, "
         "OR 10.87), hypopharynx–esophagus (lift 5.50, OR 6.77), and lip–gum (lift 9.12). "
         "Co-occurrence was strongly male-predominant (29 male-specific vs 5 female-specific "
         "enriched pairs).", False, 9.5),
    ], width=100, top=1.0, scale=180)
    img(fig, "02_associations/top_associations_lift.png", [0.04,0.40,0.92,0.30], "Figure 1. Top cancer pairs by lift and support–lift bubble plot.")
    img(fig, "01_matrix/cooccurrence_heatmap.png",        [0.04,0.06,0.92,0.30], "Figure 2. Cancers per patient (left) and co-occurrence counts among top 25 sites (right).")
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 5: Results II — NMF ──────────────────────────────────────────────
    fig = plt.figure(figsize=A4); footer(fig, 5)
    ax = fig.add_axes([0.07, 0.86, 0.86, 0.10]); ax.axis("off")
    flow(ax, [
        ("3.2  NMF cancer programs", True, 9.5),
        ("NMF (k=7) resolved seven interpretable programs. Two — aerodigestive SCC (P2) and oral "
         "cavity field (P5) — were near-exclusively male (95–97%), had the youngest onset "
         "(52–53 y), and the highest third-cancer rates (17–19%), consistent with field "
         "cancerization. P6 captured a female reproductive/breast program.", False, 9),
    ], width=104, top=1.0, scale=120)
    img(fig, "03_nmf/nmf_programs_multiprimary.png", [0.03,0.30,0.94,0.52], "Figure 3. Seven NMF cancer co-occurrence programs (multiple-primary cohort).")
    # NMF table
    axt = fig.add_axes([0.08,0.07,0.84,0.20]); axt.axis("off")
    rows = [["Program","Theme","n","≥3 cancers","Median age","Male %"],
            ["P2","Aerodigestive SCC","615","17%","53","97%"],
            ["P5","Oral cavity field","758","19%","52","95%"],
            ["P6","Female genital/breast","730","3%","56","5%"],
            ["P4","Liver/GI","766","7%","64","69%"],
            ["P1","Colorectal","310","2%","64","57%"],
            ["P7","Urological","406","5%","72","90%"]]
    t = axt.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center",
                  bbox=[0,0,1,1], colWidths=[0.11,0.34,0.10,0.16,0.16,0.13])
    t.auto_set_font_size(False); t.set_fontsize(7.5)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r==0: cell.set_facecolor(ACCENT); cell.set_text_props(color="white",fontweight="bold")
        elif rows[r][0] in ("P2","P5"): cell.set_facecolor("#ffe9d6")
        elif r%2==0: cell.set_facecolor(BG)
    axt.set_title("Table 1. NMF program clinical profiles.", fontsize=8, color=GRAY, pad=2)
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 6: Results III — DL clustering + transitions ─────────────────────
    fig = plt.figure(figsize=A4); footer(fig, 6)
    ax = fig.add_axes([0.07, 0.86, 0.86, 0.10]); ax.axis("off")
    flow(ax, [
        ("3.3  Deep-learning clusters and convergence", True, 9.5),
        ("Autoencoder + k-means (k=3) recovered three clusters matching the association-rule and "
         "NMF domains. Cluster C3 (aerodigestive SCC) had the worst mortality (68.8%) and "
         "youngest onset (55 y). 1st→2nd primary transitions confirmed bidirectional aerodigestive "
         "flow (hypopharynx↔esophagus).", False, 9),
    ], width=104, top=1.0, scale=120)
    img(fig, "04_clustering/cluster_cancer_heatmap_k3.png", [0.04,0.58,0.92,0.24], "Figure 4. Cancer prevalence by autoencoder cluster (k=3).")
    img(fig, "02_associations/cancer_transition_matrix.png",[0.06,0.27,0.88,0.27], "Figure 5. First → second primary cancer transitions.")
    axt = fig.add_axes([0.10,0.06,0.80,0.16]); axt.axis("off")
    rows = [["Cluster","Theme","n","Median age","Male %","Mortality","≥3 cancers"],
            ["C3","Aerodigestive SCC field","1,071","55","96%","68.8%","13%"],
            ["C1","GI/visceral mixed","1,827","61","76%","58.8%","11%"],
            ["C2","Female-enriched/breast","1,170","61","30%","49.1%","4%"]]
    t = axt.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center",
                  bbox=[0,0,1,0.88], colWidths=[0.10,0.30,0.10,0.15,0.11,0.13,0.12])
    t.auto_set_font_size(False); t.set_fontsize(7.5)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r==0: cell.set_facecolor(ACCENT); cell.set_text_props(color="white",fontweight="bold")
        elif rows[r][0]=="C3": cell.set_facecolor("#ffe9d6")
        elif r%2==0: cell.set_facecolor(BG)
    axt.set_title("Table 2. Autoencoder cluster clinical profiles.", fontsize=8, color=GRAY, pad=2)
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 7: Results IV — SIR + disease trajectories ───────────────────────
    fig = plt.figure(figsize=A4); footer(fig, 7)
    ax = fig.add_axes([0.07, 0.85, 0.86, 0.11]); ax.axis("off")
    flow(ax, [
        ("3.4  Second-primary risk and disease trajectories", True, 9.5),
        ("Across 288,424 person-years, 69 of 179 index→target pairs showed significantly elevated "
         "second-primary risk (FDR<0.05). The highest were confined to the aerodigestive field: "
         "esophagus→larynx (SIR 12.2, 95% CI 5.6–23.2) and esophagus→hypopharynx (SIR 12.0, "
         "95% CI 8.1–17.3). Directional trajectory analysis distinguished two modes (below).",
         False, 9),
    ], width=104, top=1.0, scale=130)
    img(fig, "05_sir_trajectories/sir_forest.png",      [0.02,0.40,0.50,0.42], "Figure 6. Second-primary SIR (index → later primary, FDR<0.05).")
    img(fig, "05_sir_trajectories/trajectory_graph.png",[0.50,0.40,0.49,0.42], "Figure 7. Directional disease trajectories (edge colour = % synchronous).")
    axt = fig.add_axes([0.08,0.07,0.84,0.26]); axt.axis("off")
    rows = [["Trajectory","Forward","Reverse","% sync.","Mode (dir. FDR)"],
            ["Hypopharynx ↔ Esophagus","33","29","60%","Field — synchronous (ns)"],
            ["Esophagus ↔ Larynx","12","9","46%","Field — mixed (ns)"],
            ["Breast → Lung","90","10","24%","Sequential (<0.001)"],
            ["Breast → Corpus uteri","51","10","10%","Sequential (<0.001)"],
            ["Colon → Liver","57","22","30%","Sequential (0.002)"],
            ["Cervix → Breast","57","22","21%","Sequential (0.002)"],
            ["Colon → Prostate","55","21","23%","Sequential (0.002)"]]
    t = axt.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center",
                  bbox=[0,0,1,0.9], colWidths=[0.32,0.13,0.13,0.12,0.30])
    t.auto_set_font_size(False); t.set_fontsize(7.5)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r==0: cell.set_facecolor(ACCENT); cell.set_text_props(color="white",fontweight="bold")
        elif r in (1,2): cell.set_facecolor("#ffe9d6")
        elif r%2==0: cell.set_facecolor(BG)
    axt.set_title("Table 3. Disease-trajectory directionality and synchronous/metachronous split.",
                  fontsize=8, color=GRAY, pad=2)
    fig.text(0.09, 0.065,
             "† Forward/Reverse = directional pair counts (metachronous only); % sync = synchronous fraction of all pairs (incl. same-day ±6 mo).\n"
             "  dir. FDR = Benjamini-Hochberg–corrected binomial directional test; ns = not significant (FDR ≥ 0.05).",
             fontsize=6.5, color=GRAY, va="top")
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 8: Discussion ────────────────────────────────────────────────────
    fig = plt.figure(figsize=A4); footer(fig, 8)
    ax = fig.add_axes([0.07, 0.05, 0.86, 0.92]); ax.axis("off")
    blocks = [
        ("4.  DISCUSSION", True, 12), ("", False, 5),
        ("In 78,578 registry patients, three orthogonal unsupervised methods — association rule "
         "mining, NMF, and deep autoencoder clustering — independently converged on the same "
         "three cancer co-occurrence domains. The dominant domain is an aerodigestive "
         "squamous-cell-carcinoma field spanning the oral cavity, pharynx, larynx, and esophagus. "
         "This domain is near-exclusively male, presents at the youngest age, carries the worst "
         "survival, and produces the highest rate of a third primary cancer.", False, 9.5),
        ("", False, 3),
        ("Methodological convergence is the central strength of this study. Each method has "
         "distinct assumptions: association rules are pairwise and model-free; NMF imposes "
         "parts-based non-negative additivity; the autoencoder captures nonlinear higher-order "
         "structure. That all three recover the same aerodigestive, female-reproductive, and "
         "visceral/urological partition argues strongly that these domains reflect shared "
         "carcinogenic biology rather than analytic artifact.", False, 9.5),
        ("", False, 3),
        ("The aerodigestive domain is the molecular-epidemiologic signature of betel quid, "
         "tobacco, and alcohol exposure in Taiwan. Standardized incidence ratios confirmed and "
         "quantified the excess risk: esophagus carried a 12-fold elevated risk of a subsequent "
         "laryngeal or hypopharyngeal primary. Crucially, trajectory analysis revealed two distinct "
         "temporal modes. Aerodigestive pairs (e.g. hypopharynx–esophagus) were bidirectional and "
         "predominantly synchronous (60% within 6 months; no significant direction), consistent "
         "with simultaneous field transformation detected at index work-up. In contrast, "
         "breast→lung, breast→corpus uteri, and colorectal trajectories were strongly "
         "unidirectional and metachronous (10–24% synchronous; FDR<0.01), reflecting sequential "
         "second primaries arising years later under surveillance. The 13–19% third-cancer rate in "
         "the aerodigestive domain greatly exceeds the cohort-wide MPC rate (5.2%), reinforcing the "
         "imperative for systematic, synchronous pan-aerodigestive panendoscopy in any patient "
         "presenting with one such SCC.", False, 9.5),
        ("", False, 3),
        ("The female reproductive/breast and visceral GI/urological domains likely reflect, "
         "respectively, shared hormonal/genetic susceptibility and age-related accumulation with "
         "surveillance-driven detection. The latter must be interpreted cautiously, as registry "
         "coding cannot fully separate metachronous primaries from metastatic spread (e.g., "
         "colon–liver).", False, 9.5),
        ("", False, 3),
        ("Limitations", True, 9.5),
        ("This single-registry study cannot distinguish same-site recurrence from true second "
         "primary with certainty, and cancer-sequence numbering may undercount later primaries. "
         "Metastasis may contaminate visceral pairs. Vital status is a registry snapshot, not a "
         "fixed-horizon survival estimate. The all-patient UMAP uses a subsample for embedding "
         "(visualization only). Finally, molecular and exposure-level data were unavailable to "
         "confirm shared mechanisms directly.", False, 9.5),
        ("", False, 3),
        ("Conclusions", True, 9.5),
        ("Convergent unsupervised learning identifies an aerodigestive field-cancerization domain "
         "as the principal driver of multiple primary cancers in Taiwan. The finding supports "
         "intensified, field-directed second-cancer surveillance and demonstrates the value of "
         "multi-method concordance for credible discovery from cancer-registry data.", False, 9.5),
    ]
    flow(ax, blocks, width=100, top=1.0, scale=560)
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 9: References ────────────────────────────────────────────────────
    fig = plt.figure(figsize=A4); footer(fig, 9)
    ax = fig.add_axes([0.07, 0.05, 0.88, 0.92]); ax.axis("off")
    ax.text(0, 1.0, "REFERENCES", transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=NAVY, va="top")
    refs = [
        "Vogt A, Schmid S, Heinimann K, et al. Multiple primary tumours: challenges and approaches, a review. ESMO Open. 2017;2(2):e000172.",
        "Coyte A, Morrison DS, McLoone P. Second primary cancer risk - the impact of applying different definitions of multiple primaries. BMC Cancer. 2014;14:272.",
        "Slaughter DP, Southwick HW, Smejkal W. Field cancerization in oral stratified squamous epithelium. Cancer. 1953;6(5):963–968.",
        "Lee CH, Lee JM, Wu DC, et al. Independent and combined effects of alcohol, tobacco and betel quid on esophageal cancer risk in Taiwan. Int J Cancer. 2005;113(3):475–482.",
        "Chuang SC, Scelo G, Tonita JM, et al. Risk of second primary cancer among patients with head and neck cancers. Int J Cancer. 2008;123(10):2390–2396.",
        "Morris LGT, Sikora AG, Patel SG, et al. Second primary cancers after an index head and neck cancer. J Clin Oncol. 2011;29(6):739–746.",
        "Alexandrov LB, Nik-Zainal S, Wedge DC, et al. Signatures of mutational processes in human cancer. Nature. 2013;500(7463):415–421.",
        "Lee DD, Seung HS. Learning the parts of objects by non-negative matrix factorization. Nature. 1999;401(6755):788–791.",
        "McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection. J Open Source Softw. 2018;3(29):861.",
        "Hinton GE, Salakhutdinov RR. Reducing the dimensionality of data with neural networks. Science. 2006;313(5786):504–507.",
        "Agrawal R, Srikant R. Fast algorithms for mining association rules. Proc 20th VLDB. 1994:487–499.",
        "Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine learning in Python. J Mach Learn Res. 2011;12:2825–2830.",
        "Chiang CJ, Wang YW, Lee WC. Taiwan's Nationwide Cancer Registry System of 40 years. J Formos Med Assoc. 2019;118(5):856–858.",
        "Curtius K, Wright NA, Graham TA. An evolutionary perspective on field cancerization. Nat Rev Cancer. 2018;18(1):19–32.",
        "Warren S, Gates O. Multiple primary malignant tumors: a survey of the literature and statistical study. Am J Cancer. 1932;16:1358–1414.",
        "Jensen AB, Moseley PL, Oprea TI, et al. Temporal disease trajectories condensed from population-wide registry data covering 6.2 million patients. Nat Commun. 2014;5:4022.",
        "Breslow NE, Day NE. Statistical Methods in Cancer Research. Vol II: The Design and Analysis of Cohort Studies. IARC Sci Publ. 1987;(82):1–406.",
    ]
    y = 0.95
    for i, r in enumerate(refs, 1):
        wrapped = tw.fill(f"{i}.  {r}", 104)
        ax.text(0, y, wrapped, transform=ax.transAxes, fontsize=8, va="top",
                color="#111111", linespacing=1.4)
        y -= (8*(wrapped.count("\n")+1)*1.55)/560
    ax.text(0, y-0.01, "Reference style: Vancouver/ICMJE. Author/affiliation, IRB approval, and "
            "data-availability statements to be completed before submission.",
            transform=ax.transAxes, fontsize=7, color=GRAY, style="italic", va="top")
    pdf.savefig(fig, dpi=200, bbox_inches="tight"); plt.close(fig)

print(f"PDF saved: {OUT}  ({OUT.stat().st_size//1024} KB)")
