# Taiwan Cancer Registry — Memo Log

Running record of sessions, decisions, findings, and open items.
One entry per session, newest at the top.

---

## 2026-06-03

**Projects completed this session:** PRs #10–13 (temporal trends, syndrome validation, sex atlas, multi-history)

### Temporal Trends (PR #10, scripts 18–20)

Pre-registered hypotheses tested against 78,619 first-primary patients 2003–2020:

| Hypothesis | Site | ρ | Verdict |
|---|---|---|---|
| H1: HBV vaccination → C22 declining | C22 | −0.983 | ✅ Confirmed |
| H2: Betel regulation → C13 declining | C13 | −0.773 | ✅ Confirmed |
| H2: C12 declining | C12 | +0.562 | ❌ Rising — unexpected |
| H3: Metabolic syndrome → C54 rising | C54 | +0.925 | ✅ Confirmed |
| H3: C50 breast rising | C50 | −0.238 ns | ❌ Flat |

Bonus: C53 cervix ρ=−0.946 (HPV), C61 prostate ρ=+0.895 (PSA). Novel-1 VAE cluster (HBV/GI) ρ=−0.977 mirrors C22 decline.

**C12 pyriform rising while C13 hypopharynx falling** is the most clinically interesting anomaly — these are anatomically adjacent betel/tobacco sites that diverge. Possible: differential endoscopic detection or anatomical reclassification.

**Birth cohort C22**: all cohorts declining; born ≥1980 cohort underpowered (n=3,413) — need registry through 2030 to test HBV vaccination cohort effect conclusively.

---

### Hereditary Syndrome Validation (PR #11, scripts 21–22)

Started from 43 actionable LFS+Cowden candidates (autoencoder Screen, Script 06).

**After phenotypic refinement:**
- LFS refined (C71 present + age<50): **0** — all 26 LFS candidates have C50 breast only, none have C71 brain (hallmark site). Effective LFS registry definition = {C50, C71} only (C74 adrenal, C49 soft tissue absent from registry).
- Cowden refined (match_score=1.0): **3** — ages 46, 49, 57yr; all female; C50+C54

**Three Cowden high-confidence candidates deserve PTEN germline testing referral.**

Registry gap note: C73 thyroid absent → can't detect thyroid arm of Cowden. C49, C74 absent → LFS non-breast components undetectable.

**PID join fix**: `all_cancers.csv` 病歷號(2) → read as float by pandas → `.astype("Int64").astype(str)` for string matching to patient_meta PIDs.

---

### Sex-Specific Atlas (PR #12, scripts 23–24)

Full sex distribution across 37 sites; M:F OR + age×sex + multi-cancer by sex + VAE axis sex separation.

Key findings:
- 16 male-dominant sites, 4 female-dominant (C50/C53/C54/C56), 17 neutral
- Strongest male: C12 OR=40×; C13 OR=29×; C06 OR=21×; C15 OR=17×
- UADT males present 5–9yr younger at same site (C06: Δ=−9yr, C02: Δ=−7yr, C15: Δ=−5yr) — earlier betel exposure onset
- Males 2× more likely to develop multi-primary cancers (6.8% vs 3.4%); excess not confined to UADT (Other-axis 2.9× excess)
- VAE z4 UADT rank-biserial=−0.621 (male); z5 Hormonal rbi=+0.409 (female)
- Survival log-rank M vs F: p<0.001 (males worse)

**Most interesting**: male multi-cancer excess is NOT explained by UADT alone. Non-UADT males 2.9× more likely than non-UADT females — suggests systemic sex difference in multi-primary risk (immune? hormonal? carcinogen synergy?).

**Note on Cox model**: Cox HR for sex was numerically unstable (HR=178 artifact from near-separation). Replaced with log-rank test. Proper stratified Cox with sex × axis interaction still needed.

---

### Multi-History Transformer (PR #13, scripts 25–26)

**Principal finding: training/inference mismatch — Script 07's BERT training does not support causal clinical inference.**

| Context k | R@1 | vs random |
|---|---|---|
| k=1 (first cancer only) | 0.067 | 2.5× |
| k=2 (two cancers) | 0.054 | 2.0× |
| k=3+ | ≤0.045 | decreasing |

Model predicts C18 (colon) as top-1 for ~40% of patients regardless of first cancer → collapsed to marginal distribution in causal mode.

**Root cause**: Script 07 uses leave-one-out BERT masking where ALL cancers (past AND future) are visible as context at masked positions. This trains bidirectional co-occurrence, not forward sequential prediction. In clinical inference, only past cancers are available → catastrophic transfer failure.

**Script 12 discrepancy**: Script 12 reported R@1=0.232 for first-only context. Script 25 gets R@1=0.067. Only 18% PID overlap between val sets (810 vs 975 patients) due to different build_sequences implementations. For the 149 overlapping patients, Script 12 reports R@1=0.22 but Script 25 computes 0.08 → same patients, different predictions. Either Script 12 used different (better-trained) model weights, or Script 12's val split leaked training patients.

**Fix**: retrain with causal (lower-triangular) attention mask (GPT-style). Each position attends only to past positions. Expected R@1 recovery to ~0.15–0.25 range.

**Architecture notes** (for future retraining):
- Checkpoint: `dl_vae/models/transformer_weights.pt`
- FF_DIM=128 (not 256), head name=`mlm_head`
- TimeEncoding: no register_buffer; computes sinusoidal encoding on-the-fly from log(1 + days/30)
- Sex input: `(B,)` scalar per sample (NOT `(B,S)`); `sex_embed(sex.long()).unsqueeze(1)` broadcasts to all positions

---

## Prior session (2026-06-02 / 2026-06-03)

**Projects completed:** PRs #1–9

| PR | Project | Key result |
|---|---|---|
| #1 | VAE axes + masked predictor | 3 active axes; UADT/hormonal/HBV; z4 male, z0/z5 female |
| #2 | Hereditary syndrome screen | 43 actionable (LFS+Cowden); MEN1=81 discarded |
| #3 | DeepHit competing risks | CIF 5yr death=57%, 2nd-cancer=10% |
| #4 | Cancer sequence Transformer | R@1=0.312 (2.4×MLP); bidirectional BERT |
| #5/#6 | UADT manuscript + peer review | TV Cox HR=2.14; ΔHR=1.28 landmark vs TV; EHR pathway; draft 19pp |
| #7 | Three-axis taxonomy | z4=UADT male; z0/z5=hormonal female; C3 HR=44 ⚠️ PH check needed |
| #8 | Sequence surveillance calendar | R@1=0.232 (but see PR #13 caveat); 74% UADT within 6mo |
| #9 | HBV/GI-systemic axis | C22 ρ=−0.983; two non-overlapping axes; SIR<<1 biologically expected |

---

## Open Issues (all projects)

### UADT manuscript
- [ ] Fill author name/affiliation placeholders
- [ ] Fill `[IRB reference number]`
- [ ] Fill `[Funding statement]`
- [ ] Regenerate figures at 300 dpi for Oral Oncology submission (currently 150 dpi)

### Three-axis taxonomy (PR #7)
- [ ] Rename cluster 3 — "UADT field" label is wrong (top sites C54/C56/C16 endometrial/uterine/stomach)
- [ ] C3 HR=44: Schoenfeld residuals to test PH assumption; consider stratified Cox or time-split model
- [ ] Revisit k: silhouette monotonically increasing — consider k=6 or k=7

### Sequence surveillance calendar (PR #8)
- [ ] Script 12 R@1=0.232 reliability uncertain (see PR #13 analysis)
- [ ] Retrain Transformer with causal masking → re-evaluate R@k
- [ ] Prospective EHR integration with endoscopic yield endpoint

### Temporal trends (PR #10)
- [ ] C12 pyriform rising mechanism: clinical review of reclassification vs surveillance artifact
- [ ] Multi-primary trend: replace FU restriction with Poisson person-year model
- [ ] Birth-cohort HBV test: revisit when registry extends to 2030+

### Syndrome validation (PR #11)
- [ ] System-wide C50+C54 screen (not just top-1% anomalous): prevalence likely underestimated
- [ ] Coordinate with registry: add C49, C73, C74 site codes for complete LFS/Cowden screening

### Sex-specific atlas (PR #12)
- [ ] Stratified Cox: sex × axis interaction term (current model numerically unstable)
- [ ] Age-standardised incidence rates (vs general population denominators)
- [ ] Multi-cancer follow-up bias: Poisson person-year model

### Multi-history Transformer (PR #13)
- [ ] Retrain with causal masking (GPT-style lower-triangular attention)
- [ ] Re-evaluate R@k on proper causal val set
- [ ] External validation on SEER multi-primary data

---

## Data and Security

**NEVER commit to git:**
- `data/processed/all_cancers.csv` (PIDs + patient data)
- `dl_vae/data/` (derived embeddings, matrices)
- `dl_vae/models/` (trained weights)
- `uadt_field/data/`
- Any file with `pid`, `病歷號`, or patient-level identifiers

These are enforced in `.gitignore`. All result CSVs that reach git are aggregated (site-level, cluster-level, etc.) — no patient rows.

---

## Key Constants (never change mid-project)

| Constant | Value | Used in |
|---|---|---|
| `FIELD_SITES` | C02/C03/C04/C05/C06/C09/C10/C12/C13/C15 | uadt_field |
| `SYNC_MO` | 6 months | uadt_field trajectory |
| `LANDMARK_MO` | 6 months | uadt_field survival |
| `KMEANS_K` | 5 | dl_vae clusters |
| `KMEANS_SEED` | 42 | dl_vae clusters |
| `SEED` / `VAL_FRAC` | 42 / 0.20 | Transformer val split |
| `MAX_SEQ` | 8 | Transformer sequence length |
| `PAD/MASK/CLS` | 0 / 1 / 2 | Transformer vocab |
| `SITE_OFFSET` | 3 | Transformer site tokens start at 3 |
| `D_MODEL / N_HEAD / N_LAYERS / FF_DIM` | 64 / 4 / 2 / 128 | Transformer architecture |

---

## Draft PDFs on disk (not committed)

| File | Size | Pages | Status |
|---|---|---|---|
| `uadt_field/manuscript/uadt_manuscript.pdf` | 1.35 MB | 19 | Pending author placeholders |
| `dl_vae/results/VAE_Axes_Draft.pdf` | — | 7 | Done |
| `dl_vae/results/Syndrome_Screen_Draft.pdf` | — | 4 | Done (superseded by Validation draft) |
| `dl_vae/results/Taxonomy_Draft.pdf` | — | 8 | C3 label ⚠️ |
| `dl_vae/results/Surveillance_Draft.pdf` | — | 8 | R@1 claim uncertain |
| `dl_vae/results/HBV_Axis_Draft.pdf` | — | 8 | Done |
| `dl_vae/results/Temporal_Draft.pdf` | 395 KB | 8 | Done |
| `dl_vae/results/Syndrome_Validation_Draft.pdf` | 212 KB | 8 | Done |
| `dl_vae/results/Sex_Atlas_Draft.pdf` | 379 KB | 8 | Done |
| `dl_vae/results/MultiHistory_Draft.pdf` | 271 KB | 8 | Done — negative result |
