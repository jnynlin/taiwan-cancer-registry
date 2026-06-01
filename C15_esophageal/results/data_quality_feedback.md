# Data Quality Feedback — Taiwan Cancer Registry C15 Esophageal Cancer Cohort
**Cohort:** 2,367 cases · ICD-10 C15 · 2006–2020  
**Prepared:** 2026-05-31  
**Purpose:** Structured gap analysis for registry managers, data custodians, and future data-linkage planning

---

## Executive Summary

The registry captures core oncological variables (vital status, histology, gross staging, treatment
presence/absence) at near-complete rates, which enables survival and descriptive analyses. However,
**treatment dose and regimen details are systematically absent**, recurrence data are entirely empty,
and staging completeness collapses when pathologic confirmation is unavailable. These gaps limit
the registry to hypothesis-generating findings and preclude dose-response, regimen-comparative,
or recurrence analyses without supplementary data linkage.

---

## 1. Chemotherapy — Critical Gaps

| Variable | Completeness | Issue |
|---|---|---|
| Regimen code `(163)` | 100% | **Categorical only** (0/1/2/3/multi-agent+targeted) — no drug names |
| Chemo method `(160)` | 73% | Single vs multi-agent only — no drug identity |
| **Cycles `(161)`** | **5.3% non-zero** | **Effectively absent** — 94.7% are 0 or missing |
| Chemo start date | 82% | Sufficient for timing analysis |
| CCRT flag `(159)` | 100% | Binary only — no concurrent drug identified |

### What is missing
- **Drug names are not captured anywhere** — cisplatin, carboplatin, 5-FU, paclitaxel,
  oxaliplatin are all collapsed into a single-digit regimen code. This makes regimen-comparative
  analysis (e.g., cisplatin-5FU vs carboplatin-paclitaxel for CCRT) impossible.
- **Dose per cycle (mg/m²)** — not captured in any field.
- **Cumulative dose** — only a cycle count proxy exists, and it is populated for only 126 cases
  (5.3%). Meaningful dose-response analysis is not feasible from this dataset alone.
- **Dose reductions, delays, or early termination** — not recorded.
- **Response assessment** — pathologic complete response rate after neoadjuvant therapy is absent.

### Recommendation
Link registry records to the hospital pharmacy dispensing system or NHI reimbursement claims
(健保申報資料) to retrieve: drug name, dose per cycle, number of cycles, start/end dates, and
dose modifications. The NHI claims database has drug-level data for all reimbursed regimens.

---

## 2. Radiation Therapy — Partially Captured, Structurally Encoded

| Variable | Completeness | Issue |
|---|---|---|
| RT start / end date | 100% (if RT given) | Dates reliable |
| **RT dose (Gy) `(146)`** | **50% of cohort** | Stored as cGy integer (5040 = 50.4 Gy); 658 cases have dose=0 |
| RT fractions `(147)` | 72% | Median 33 fx (range 1–99); 1 or 2 fractions likely data entry errors |
| RT technique `(144)` | 100% populated | **Code 4 = IMRT, 0 = none/unknown, 999 = not recorded** — but code dictionary not standardized across form versions |
| Lower RT dose field `(149)` | 47% | For boost or nodal volumes; inconsistently recorded |
| Brachytherapy `(134)` | **0%** | Field present but entirely empty for this cohort |

### Issues
- **Dose stored in cGy without explicit unit labeling** — value 5040 means 50.40 Gy (CCRT
  standard), but 180 could be 1.8 Gy (one fraction) or a data entry error.
- **RT technique codes vary across form versions** (A→H versions) and are not harmonized.
  Code "4" likely = IMRT in later versions but interpretation is ambiguous in earlier sheets.
- **Fractionation scheme** (conventional 1.8–2.0 Gy/fx vs hypofractionation) is inferrable
  from dose÷fractions but not explicitly coded.
- **Target volume** (primary, nodal, elective nodal irradiation) is not itemized.

### Recommendation
- Add unit clarification (cGy vs Gy) to data dictionary.
- Standardize RT technique code dictionary across all long-form versions.
- Add explicit field for: total prescribed dose (Gy), dose per fraction (Gy), technique
  (IMRT/3DCRT/VMAT), and concurrent systemic agent.
- For existing data: compute dose in Gy as `dose_field / 100` where value > 500, flag
  values < 20 Gy as likely errors.

---

## 3. Staging — Pathologic Stage Nearly Unusable

| Variable | Completeness | Issue |
|---|---|---|
| Clinical stage `(95)` | 76.7% real values | 23.3% coded as 999/888/unknown |
| **Pathologic stage `(101)`** | **28.3% real values** | **71.7% = 888/999 (not assessed / N/A)** |
| Clinical T/N/M | 73% each | Sufficient for sub-analysis |
| AJCC edition `(104)` | 73% | Mixed 6th, 7th editions — staging not comparable across years |

### Interpretation
Pathologic stage is available for only 671/2,367 cases (28.3%), which are the surgically
resected patients. The remaining 71.7% were treated with definitive CCRT, palliative
chemotherapy, or were inoperable — these receive 888 (not applicable) or 999 (unknown)
coding. **This is not a data quality error but a clinical reality**: most esophageal cancer
patients in this cohort are not resected. However, it means:
- Stage-specific survival analysis must rely on clinical staging (73% complete).
- Mixed AJCC editions (6th vs 7th) affect T and N sub-classifications — the 7th edition
  redefined M1a/M1b; direct comparison requires edition-stratified analysis.

### Recommendation
- Add AJCC edition to all stage-specific analyses as a covariate.
- For unresected patients, supplement with response evaluation imaging data (CT/PET-CT)
  from hospital PACS to capture restaging after induction therapy.

---

## 4. Recurrence — Entirely Absent

| Variable | Completeness | Status |
|---|---|---|
| First recurrence date `(76)` | **0%** | Field empty across all 2,367 cases |
| First recurrence site `(78)` | **0%** | Field empty |
| Post-recurrence treatment `(79)` | 73% populated | Coded but uninterpretable without recurrence date |

### Impact
- **Disease-free survival (DFS) and recurrence-free survival (RFS) cannot be computed.**
- Locoregional vs distant recurrence rates — unavailable.
- Time to recurrence and second-line treatment patterns — unavailable.

### Recommendation
This is the single highest-impact gap for surgical outcome research. Options:
1. **Retrospective chart review** of operated patients (n=534) to extract recurrence dates
   from follow-up clinic notes or imaging reports.
2. **NHI database linkage** — subsequent treatment codes after initial therapy can serve
   as a recurrence proxy (e.g., second-line chemotherapy, palliative RT).
3. **Prospective registry amendment** — add mandatory recurrence fields to the data entry
   workflow from 2025 onwards.

---

## 5. Lifestyle Variables — Moderate Completeness, Sentinel Inflation

| Variable | True non-missing | Issue |
|---|---|---|
| Daily cigarettes | 50% | Sentinel 98 (refused) and 99 (unknown) remove ~50% |
| Smoking years | 45% | High sentinel rate |
| Betel nut quantity | 35% | Only available in newer form versions |
| Betel nut years | 40% | Partially captured |
| Alcohol behavior | 49% | Code 9 (unknown) inflates missing |
| BMI (derived) | 59% | Height/weight sentinel 999 = not measured |
| **Smoking pack-years** | **Not calculable** | `cigarettes/day × smoking_years / 20` possible but limited coverage |

### Recommendation
- Compute pack-years for completeness-eligible cases and add as a derived variable.
- Consolidate "refused to answer" (98) vs "not recorded" (99) vs "not applicable" (0)
  into a unified missing indicator — currently ambiguous.
- For lifestyle analysis, restrict to the 1,937-case subset with newer form versions
  where lifestyle fields are more complete.

---

## 6. Structural Issues

### 6a. Multi-version Long-Form Sheets (A→H)
The file contains **6 incompatible sheet versions** with different column sets (98–160 columns):
- Columns added over time: lifestyle fields (A3–A5), MIS field (B8), margin distance (B9),
  immunotherapy dates (A23), targeted therapy (A25–A27)
- **Columns present in all versions**: demographics, histology, surgery type, margin status,
  LN counts, vital status, death cause
- **Consequence**: pooling all sheets creates a 262-column sparse matrix where 40–60% of
  newer fields are structurally absent for early-era cases (not truly missing — never collected)

### 6b. Date Format Inconsistency
- Dates stored as 6-digit integers when Excel drops leading zero: `920528` instead of `0920528`
- Day/month coded as `99` (unknown specific date) is valid in the registry but breaks
  standard date parsing — requires imputation to first of month/year
- Recommendation: Enforce 7-digit zero-padded string format in future submissions

### 6c. Sentinel Value Ambiguity
Multiple sentinel values overlap in meaning:
- `0` = not performed OR not applicable OR genuinely zero
- `9`, `99`, `999` = unknown/not recorded
- `88`, `888`, `8888888` = not applicable (field irrelevant for this patient)
- `98` = refused to answer (lifestyle fields)
- Recommendation: Add a standardized data dictionary sheet to the registry file

---

## 7. Priority Recommendations for Data Enhancement

| Priority | Action | Expected impact |
|---|---|---|
| 🔴 High | Link to NHI pharmacy claims for drug names + cycle counts | Enables regimen-comparative and dose-response analysis |
| 🔴 High | Retrospective chart review for recurrence dates (n=534 operated) | Enables DFS, RFS, and recurrence pattern analysis |
| 🟡 Medium | Standardize RT dose field unit (cGy → Gy) + validate outliers | RT dose-response and fractionation analysis |
| 🟡 Medium | Add pathologic CR rate field for neoadjuvant cases | CROSS-trial-comparable outcomes |
| 🟡 Medium | Prospectively add ECOG/KPS performance status | Confounder adjustment in survival models |
| 🟢 Lower | Add PD-L1 / HER2 molecular marker fields | Biomarker-stratified immunotherapy analysis |
| 🟢 Lower | Encode tumor length and circumferential involvement | Endoscopic staging sub-analysis |

---

## 8. What IS Reliable for Publication

Despite the gaps above, the following analyses are well-supported by the current data:

✅ **Incidence trends** (2006–2020) — complete  
✅ **Overall survival** by stage, histology, subsite, sex — complete  
✅ **Surgery type and R0 margin impact on OS** — well-supported (76–100% complete)  
✅ **CCRT vs non-CCRT survival benefit** — 100% complete flag  
✅ **LN dissection extent and LN ratio** — adequate (99% LN counts)  
✅ **Perioperative treatment sequencing** (neoadjuvant vs adjuvant) — 100%  
✅ **Deep learning patient subtype discovery** — complete on available features  

⚠️ **Presented with caveats in manuscript:**  
- Chemo dose-response (cycle count 5.3% — hypothesis-generating only)  
- RT dose analysis (50% — exploratory only)  
- Lifestyle risk factor analysis (35–50% — subset analysis)  

❌ **Not feasible without additional data:**  
- Regimen-comparative survival (cisplatin vs carboplatin)  
- Disease-free survival / recurrence-free survival  
- Pathologic complete response rate after neoadjuvant CCRT  
- Molecular biomarker stratification  

---

*This feedback document was generated from systematic completeness analysis of all 262 registry  
variables across 2,367 C15 esophageal cancer cases. Recommendations are grounded in the specific  
gaps identified and aligned with standards from ESMO [ref 18], NCI SEER, and Taiwan HPA cancer  
registry reporting guidelines.*
