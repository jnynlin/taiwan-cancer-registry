# Co-existing Cancer Pattern Discovery — Results
**Taiwan Cancer Registry 2006–2020 · Unsupervised Multi-Label Learning**
**Cohort:** 78,578 patients · 84,157 diagnoses · 46 cancer site codes (ICD-O C-codes, 3-char)

---

## 1. Cohort & Multi-Primary Landscape

| Metric | Value |
|---|---|
| Unique patients | 78,578 |
| Total cancer diagnoses | 84,157 |
| Patients with ≥2 distinct cancer types | 4,068 (5.2%) |
| Patients with ≥3 | 536 (0.7%) |
| Patients with ≥4 | 86 |
| Max distinct cancers per patient | 7 |

Most prevalent sites among multi-primary patients: Breast, Lung, Colon, Liver,
**Esophagus**, Oral cavity (Mouth/Tongue/Gum).

---

## 2. Association Rule Mining (pairwise lift / odds ratio)

**Top co-occurring cancer pairs** (all p≪0.001; lift > 1 = co-occur more than chance):

| Cancer A | Cancer B | n co-occur | Lift | Odds Ratio |
|---|---|---|---|---|
| Lip | Mouth NOS | 65 | 6.48 | 8.15 |
| Base of tongue | Pyriform sinus | 13 | 6.29 | 6.65 |
| Gum | Palate | 31 | 5.99 | 6.53 |
| **Hypopharynx** | **Esophagus** | **156** | **5.50** | **6.77** |
| Base of tongue | Tonsil | 14 | 4.58 | 4.81 |
| Gum | Mouth NOS | 114 | 4.32 | 5.06 |
| Esophagus | Larynx | 39 | 3.69 | 4.08 |
| Oropharynx | Esophagus | 40 | 3.40 | 3.72 |

**Sex stratification:** 29 male-specific high-lift pairs (all aerodigestive: oral ↔
esophagus/hypopharynx/larynx) vs only 5 female-specific. Cancer co-occurrence in this
registry is overwhelmingly a **male aerodigestive field-cancerization phenomenon**,
consistent with betel nut + tobacco + alcohol synergy.

Outputs: `results/02_associations/` — `association_rules_{all,sig,male,female,early_onset,late_onset}.csv`,
`lift_heatmap.png`, `top_associations_lift.png`

---

## 3. NMF Cancer Co-occurrence Programs (k=7, multi-primary cohort)

Non-negative matrix factorization decomposed the multi-primary patient × cancer matrix
into 7 latent "cancer programs" (analogous to mutational signatures):

| Program | Biological theme | n | ≥3 cancers | Median age | Male % | Top sites (loading) |
|---|---|---|---|---|---|---|
| P1 | Colorectal | 310 | 2% | 64 | 57% | Colon (15.4), Rectum (4.0), Stomach (1.0) |
| **P2** | **Aerodigestive SCC** | 615 | **17%** | 53 | **97%** | Esophagus (5.9), Hypopharynx (1.9), Pyriform (1.3) |
| P3 | Lung-dominated | 483 | 8% | 64 | 74% | Lung (4.9) |
| P4 | Liver / GI | 766 | 7% | 64 | 69% | Liver (4.3), Stomach (0.2) |
| **P5** | **Oral cavity field** | 758 | **19%** | 52 | **95%** | Mouth (3.2), Tongue (2.7), Gum (1.3), Palate (0.7) |
| P6 | Female genital / breast | 730 | 3% | 56 | 5% | Breast (4.4), Cervix (1.1), Corpus uteri (0.7) |
| P7 | Urological | 406 | 5% | 72 | 90% | Prostate (3.9), Bladder (2.3) |

**Key signal:** Programs **P2 + P5** (aerodigestive + oral cavity) are nearly all male
(95–97%), have the youngest onset (52–53 yr), and the **highest rate of a 3rd primary
cancer (17–19%)** — the hallmark of field cancerization across the upper aerodigestive
tract. P5 has the longest inter-cancer span (median 3 yr), consistent with slow
multifocal progression across oral subsites.

Outputs: `results/03_nmf/` — `nmf_programs_multiprimary.png`,
`component_weights_multiprimary.csv`, `patient_nmf_loadings_multiprimary.csv`

---

## 4. Cancer Transition Matrix (1st → 2nd Primary)

Among 4,629 patients with a sequenced 2nd primary (distinct site):

| 1st Primary | 2nd Primary | n | Interpretation |
|---|---|---|---|
| Breast | Lung | 97 | Surveillance-detected metachronous |
| **Hypopharynx** | **Esophagus** | 70 | SCC field, downward aerodigestive spread |
| Cervix uteri | Breast | 66 | Female multi-cancer susceptibility |
| Colon | Liver | 65 | (screen for metastasis vs true 2nd primary) |
| Tongue | Mouth NOS | 64 | Oral cavity field |
| Colon | Rectum / Prostate | 59 each | Colorectal & age-related male |
| **Pyriform sinus** | **Esophagus** | 54 | Pharyngeal → esophageal SCC |
| **Esophagus** | **Hypopharynx** | 52 | Bidirectional aerodigestive SCC |
| Bladder | Prostate | 49 | Urological field |

Outputs: `results/02_associations/cancer_transition_matrix.png`

---

## 5. Deep Learning: Autoencoder + UMAP + k-means

A BCE autoencoder (latent dim = 12) was trained on the multi-primary cohort and used to
embed all patients; k-means on the latent space (best k=3 by cosine silhouette) recovered
**three biologically coherent clusters — independently converging on the same domains
identified by NMF and association rules:**

| Cluster | n | Dominant cancers (mean prevalence) | Median age | Male % | Mortality | ≥3 cancers | Theme |
|---|---|---|---|---|---|---|---|
| **C1** | 1,827 | Colon (0.30) + broad GI (lung, liver, stomach) | 61 | 76% | 58.8% | 11% | Visceral / GI mixed |
| **C2** | 1,170 | **Breast (0.51)** + Liver (0.43) + Lung (0.23) | 61 | 30% | **49.1%** | 4% | Female-enriched / breast |
| **C3** | 1,071 | **Esophagus (0.56)** + Mouth (0.26) + Tongue (0.22) + Hypopharynx (0.19) | **55** | **96%** | **68.8%** | **13%** | **Aerodigestive SCC field** |

C3 is the unambiguous Taiwan betel-nut/alcohol/tobacco aerodigestive SCC signature, and
carries the **worst prognosis** (68.8% mortality), **youngest onset** (median 55 yr), and
**highest multi-cancer burden** (13% develop a 3rd primary) — the clinical fingerprint of
aggressive field cancerization. The female-enriched cluster C2 has the best survival
(49.1% mortality), consistent with breast cancer's more favorable prognosis.

The convergence of three independent unsupervised methods (association rules, NMF,
deep autoencoder clustering) on the same aerodigestive / female-reproductive / visceral-GI
partition is strong evidence these co-occurrence domains are real, not artifacts.

Outputs: `results/04_clustering/` — `umap_{n_cancers,nmf_programs,kmeans_k3}.png`,
`cluster_cancer_heatmap_k3.png`, `cluster_k3_profile.csv`, `patients_annotated_k3.csv`

---

## 6. Three Convergent Co-occurrence Domains

1. **Aerodigestive SCC field cancerization** (NMF P2+P5; AE cluster C3; 29 male association pairs)
   — esophagus + oral cavity + hypopharynx + larynx; male 95–97%; youngest onset;
   highest 3rd-cancer risk. *Taiwan-specific, lifestyle-driven.*
2. **Female reproductive tract + breast** (NMF P6; AE cluster C2)
   — breast ↔ cervix ↔ corpus uteri ↔ ovary; hormonal/genetic susceptibility.
3. **Age-related visceral / GI / urological** (NMF P1/P3/P4/P7; AE cluster C1)
   — colorectal, liver, lung, prostate, bladder; older, both sexes; surveillance-driven.

---

## Limitations
- **Same-site recurrence vs true 2nd primary** not distinguishable from registry codes alone.
- **Metastasis vs metachronous primary** — pairs like Colon→Liver may include misclassified mets.
- Cancer sequence field (癌症發生順序) may undercode later primaries.
- UMAP for all-patient view uses a 12k subsample + transform projection (visual only; clustering uses full latent space).
- Vital-status–based mortality is reported per cluster but is a registry snapshot, not a fixed-horizon survival.

## Reproduce
```bash
bash coexist_cancers/analysis/run_all.sh
```
Pipeline: 01 matrix → 02 association rules → 03 NMF → 04 autoencoder clustering.
