# Session Log — C15 Esophageal Cancer Analysis
**Dataset:** Taiwan Cancer Registry 2006–2020, single-center long-form (92–109年長表資料)  
**Cohort:** 2,367 C15 esophageal cancer cases  
**Working directory:** `/home/jnynlin/coding/taiwan-cancer-registry/C15_esophageal/`

---

## 2026-05-31 — Initial build & full analysis pipeline

### Data ingestion
- Source file: `Downloads/92-109年全癌資料(長表資料)_20220726.xlsx` (password-protected, pw=3566)
- File format: CDFV2 encrypted Excel; decrypted with `msoffcrypto-tool`
- 6 long-form version sheets (A→H), non-overlapping: 8,785 + 15,121 + 4,113 + 36,875 + 12,422 + 6,845 rows
- C15 extraction: **2,367 cases** across all sheets
- **Key date parsing fix:** Excel drops leading zero in ROC dates (e.g., `0920528` → integer `920528`); fixed with `.zfill(7)`. Day/month `99` imputed to `01`.
- **Vital status fix:** Taiwan registry codes `0 = Dead`, `1 = Alive` (opposite of standard); corrected after initial run showed events=0 in survival analysis.

### Pipeline built (scripts 01–07)
| Script | Purpose | Key output |
|---|---|---|
| `01_extract.py` | Extract C15, derive dates, vital status, OS | `data/c15_all.csv` (2,367 × 262) |
| `02_descriptive.py` | Table 1, 7 figures, treatment combinations | `results/02_descriptive/` |
| `03_survival.py` | KM curves (8 groups), Cox model, CSS | `results/03_survival/` |
| `04_deep_learning.py` | Autoencoder → UMAP → k-means → DeepSurv | `results/04_deep_learning/` |
| `05_summary_pdf.py` | 5-page brief summary PDF | `results/C15_Esophageal_Cancer_Summary.pdf` |
| `06_chemo_surgery_impact.py` | Chemo regimen/timing/cycles, surgery subgroups | `results/06_chemo_surgery/` |
| `07_paper_pdf.py` | 12-page near-submission paper (300 DPI) | `results/C15_Paper_Draft.pdf` |

---

## Key findings

### Clinical characteristics
- **n=2,367** cases; male 94.5%; median age 57 (IQR 50–64); BMI 21.5
- **SCC 85.4%**, Carcinoma NOS 7.8%, Adenocarcinoma 2.3%
- Stage III 26.4%, IV 24.3%, unknown 25.7%
- Incidence rising trend 2006→2017, plateauing 2018–2020
- Lifestyle (subset available): smoker 42%, betel nut 9.9%, alcohol regular 15.9%

### Survival
- **Mortality 77.4%** (1,832/2,367); median OS 13.4 months
- Cox C-index **0.705** (bootstrap 95% CI 0.694–0.719, B=1000)
- Strongest protective factors (multivariable):
  - Endoscopic resection: **HR 0.47** (0.38–0.59, p<0.001)
  - Radical esophagectomy: **HR 0.61** (0.52–0.72, p<0.001)
  - R0 margin: **HR 0.59** (0.52–0.68, p<0.001)
  - CCRT: **HR 0.76** (0.67–0.85, p<0.001)

### Surgery & treatment sequence (n=531 operated)
| Sequence | n | % | Median OS |
|---|---|---|---|
| Neoadjuvant → OP (any) | 9 | 1.7% | 17.9 mo |
| — Neoadjuvant CCRT → OP | 8 | 1.5% | 18.3 mo |
| OP only | 267 | 50.3% | 29.9 mo |
| OP → Adjuvant (any) | 255 | 48.0% | 25.6 mo |
| — OP → Adj RT | 130 | 24.5% | 28.2 mo |
| — OP → Adj chemo | 103 | 19.4% | 24.4 mo |
| — OP → Adj CCRT | 22 | 4.1% | 20.3 mo |

**Note:** Neoadjuvant rate (1.7%) likely underestimated — registry field `放射治療與手術順序(140)` captures RT-surgery relationship only; neoadjuvant chemo without RT may be miscoded as 0.

### Deep learning (unsupervised)
- Autoencoder (latent_dim=8, MSE=0.21) → UMAP → k-means k=3 (silhouette=0.51)
- **Cluster 1** (n=2,029): Mainstream SCC, all subsites, median OS 13.3 mo
- **Cluster 2** (n=153): Adenocarcinoma-enriched (35%), overlapping/abdominal — possible GEJ misclassification
- **Cluster 3** (n=185): Cervical/upper esophagus (37% cervical), mixed histology
- DeepSurv top features: Surgery (any) > AJCC stage > BMI > Chemotherapy > SCC histology

### Chemo & RT impact
- CCRT rate: 947/2,367 (40%); independently protective HR 0.76 (p<0.001)
- Chemo cycles: only 126/2,367 (5.3%) with non-zero cycle data — dose-response underpowered
- RT dose (where recorded): median 50.4 Gy (5040 cGy), 33 fractions
- No drug names captured in registry (cisplatin/carboplatin/5-FU not distinguishable)

---

## Manuscript deliverables
| File | Description |
|---|---|
| `results/C15_Paper_Draft.pdf` | 12-page near-submission (300 DPI); target journal: *Cancers* (MDPI) |
| `results/C15_Esophageal_Cancer_Summary.pdf` | 5-page brief summary |
| `results/cover_letter_Cancers.md` | Cover letter template (fill: author, IRB no., reviewers) |
| `results/data_quality_feedback.md` | Structured gap analysis for registry managers |

### Pending before submission
- [ ] Fill author names / institution / IRB approval no. in title page and cover letter
- [ ] Verify Table 1 age/BMI p-values (currently hardcoded; run sex-stratified Mann-Whitney)
- [ ] Supplement with NHI pharmacy data for drug-level chemo analysis
- [ ] Retrospective chart review for recurrence dates (n=534 operated)
- [ ] Consider AJCC edition stratification for staging analyses (6th vs 7th edition mixed)

---

## Known data quality issues
| Issue | Impact | Action |
|---|---|---|
| No drug names for chemo | Cannot compare cisplatin vs carboplatin regimens | Link NHI claims |
| Chemo cycles 5.3% non-zero | Dose-response infeasible | Prospective registry fix |
| Recurrence fields 0% populated | No DFS/RFS possible | Chart review for operated |
| Pathologic stage 71.7% = 888/999 | Most non-operated cases unresected (correct) | Use clinical stage |
| Neoadjuvant likely undercoded | True neoadjuvant rate unknown | Date crosscheck |
| AJCC edition mixed (6th/7th) | T/N sub-classification not comparable pre/post 2010 | Edition covariate |
| RT dose stored as cGy integer | 658 cases have dose=0 (no RT or not recorded) | Unit standardize |
| Multi-version sheets (A→H) | 40–60% of newer fields absent in early-era cases | Version-aware analysis |

---

## Environment
- Python 3.11 · pandas · lifelines · torch 2.5.1 (CPU) · umap-learn · sklearn 1.8.0
- Raw data: password-protected CDFV2 Excel (msoffcrypto-tool for decryption)
- Analysis run time: ~8 min full pipeline (steps 01–07) on CPU

---

## 2026-05-31 — New sub-project: Co-existing Cancer Patterns

### Sub-project: `coexist_cancers/`
**Goal:** Unsupervised multi-label learning to discover co-existing cancer patterns across 78,621 patients (all cancer types, full registry).

### Data scope
- Full registry: 84,157 diagnoses, 78,621 unique patients, 46 cancer site codes
- Multi-primary patients: 4,068 (5.2%) with ≥2 distinct cancer types; 536 (0.7%) with ≥3
- Patients with max 7 cancer types

### Pipeline
| Script | Method | Key output |
|---|---|---|
| `01_build_patient_matrix.py` | Multi-hot encoding (patient × cancer) | `data/patient_cancer_matrix.csv` |
| `02_association_rules.py` | Pairwise lift/OR/support + sex+age stratified | `results/02_associations/` |
| `03_nmf_patterns.py` | NMF on multi-primary cohort (k=7) | `results/03_nmf/` |
| `04_autoencoder_clustering.py` | BCE autoencoder → UMAP → k-means | `results/04_clustering/` |

### Key findings

#### Association rules (top pairs by lift)
- Hypopharynx ↔ Esophagus: n=156, **lift=5.50, OR=6.77** — strongest aerodigestive SCC field
- Oral cavity pairs (lip/gum/palate): lift 4–6 — betel nut field cancerization
- Esophagus ↔ Larynx: lift=3.69 — shared SCC exposure
- Male-specific: 29 high-lift pairs (all aerodigestive); Female-specific: only 5

#### NMF Programs (k=7, multi-primary cohort only)
| Program | Biology | n | ≥3 cancer% | Median age | Male% |
|---|---|---|---|---|---|
| P1 | Colorectal cluster | 310 | 2% | 64 | 57% |
| **P2** | **Aerodigestive SCC (esoph+pharynx)** | **615** | **17%** | **53** | **97%** |
| P3 | Lung-dominated | 483 | 8% | 64 | 74% |
| P4 | Liver/GI | 766 | 7% | 64 | 69% |
| **P5** | **Oral cavity field cancerization** | **758** | **19%** | **52** | **95%** |
| P6 | Female genital/breast | 730 | 3% | 56 | 5% |
| P7 | Urological (prostate+bladder) | 406 | 5% | 72 | 90% |

#### Cancer transition (1st → 2nd primary, top patterns)
- Hypopharynx → Esophagus: n=70 (SCC downward spread)
- Pyriform sinus → Esophagus: n=54
- Esophagus → Hypopharynx: n=52 (bidirectional field)
- Oral tongue → Mouth NOS: n=64 (oral field spread)
- Breast → Lung: n=97 (surveillance-detected)

#### Three co-occurrence domains identified
1. **Aerodigestive SCC field** (P2+P5): Male 95–97%, age 52–53, 3rd-cancer rate 17–19%, driven by betel nut/tobacco/alcohol — **Taiwan-specific pattern**
2. **Female reproductive tract + breast** (P6): Hormonal/genetic susceptibility cluster
3. **Age-related GI/urological** (P1, P4, P7): Both sexes, older, longer latency

### Known limitations / data quality notes
- Vital status (death%) unreliable in patient_meta (see C15 SESSION_LOG data quality section)
- Cancer sequence field (癌症發生順序) allows up to 7 primaries but may be undercoded for later primaries
- Same-site recurrence vs true second primary not distinguishable from registry codes alone
- Date-based sequencing (1st→2nd transition) uses registry cancer sequence number, not diagnosis dates
