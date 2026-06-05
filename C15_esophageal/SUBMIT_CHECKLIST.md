# Submission Checklist — C15 Esophageal Cancer
## Target: Cancers (MDPI) — Section: Gastrointestinal Cancers

---

## STEP 1 — Fill placeholders in manuscript
**File:** `manuscript/c15_manuscript.tex`

| Line | Placeholder | Fill with |
|------|-------------|-----------|
| 59 | `[Author 1 Name]` | First author full name |
| 60 | `[Author 2 Name]` | Second author full name (affil 1+2) |
| 61 | `[Author 3 Name]` | Third author full name (corresponding) |
| 66 | `[Department of Surgery / Oncology]` | Actual department name |
| 67 | `[Institution Name], [City]` | Hospital + city |
| 68 | `[Second Affiliation, if applicable]` | Delete line if Author 2 has only one affiliation |
| 69 | `[email@institution.edu.tw]` | Corresponding author email |
| 70 | `+886-X-XXXX-XXXX` | Corresponding author telephone |
| 74–75 | `[Date]` × 3 | Received / Accepted / Published — MDPI fills these post-acceptance; **delete for initial submission** |
| 78 | `[Standard MDPI disclaimer]` | Standard MDPI publisher's note — **delete for initial submission**; MDPI adds post-acceptance |
| 286 | `[Institution]` | IRB-issuing institution |
| 286 | `[IRB-XXXX-XXXX]` | IRB approval number (first occurrence) |
| 670 | `[Author 1]` | First author name (CRediT contributions) |
| 672 | `[Author 2]` | Second author name |
| 673 | `[Author 3]` | Third author name |
| 682 | `[Institution]` | IRB institution (second occurrence) |
| 683 | `[IRB-XXXX-XXXX]` | IRB approval number (second occurrence) |

> Pre-written CRediT roles (lines 670–673):
> - Author 1: Conceptualisation, Methodology, Software, Formal analysis, Writing — original draft
> - Author 2: Data curation, Validation
> - Author 3: Supervision, Writing — review and editing
>
> Adjust if these do not reflect actual contributions.

---

## STEP 2 — Fill placeholders in cover letter
**File:** `results/cover_letter_Cancers.md`

| Line | Placeholder | Fill with |
|------|-------------|-----------|
| 1 | `[Corresponding Author Name, MD/PhD]` | Name + degree |
| 2 | `[Division of Gastroenterology / ...]` | Actual division/department |
| 3 | `[Institution Name]` | Hospital / university |
| 4 | `[Address Line 1]` | Street address |
| 5 | `[City, Country, Postal Code]` | e.g. Taichung, Taiwan 404 |
| 6 | `[email@institution.edu.tw]` + phone | Email + +886-X-XXXX-XXXX |
| 54 | `[Institution Name]` | IRB institution |
| 54 | `[IRB-XXXX-XXXX]` | IRB approval number |
| 56 | `[XXXX]` | Registry data release agreement number |
| 67 | `[Corresponding Author Name, MD/PhD]` | Sign-off: name + degree |
| 68 | `[Title, Department]` | e.g. Associate Professor, Dept of Oncology |
| 69 | `[Institution]` | Hospital / university |
| 70 | `[Email]` | Email |
| 76 | `[Reviewer 1 Name], [Institution]` | Suggested reviewer 1 — esophageal surgery/outcomes |
| 77 | `[Reviewer 2 Name], [Institution]` | Suggested reviewer 2 — GI oncology registry |
| 78 | `[Reviewer 3 Name], [Institution]` | Suggested reviewer 3 — DL for clinical oncology |

> ⚠ Cover letter says "single-center Taiwan Cancer Registry" — verify this is accurate.
> The national TCR captures all hospitals; if this is truly national data, change to
> "population-based Taiwan Cancer Registry" to avoid a reviewer challenge.
>
> ⚠ Cover letter claims "one of the largest real-world Asian esophageal cancer cohorts
> reported to date" — Cheng et al. 2018 (Cancer Medicine) reported n=14,394 from the
> same registry. Remove or qualify this claim.

---

## STEP 3 — MDPI-specific requirements
**Cancers (MDPI)** has specific formatting requirements different from Elsevier:

- [ ] **Article processing charge (APC)**: Cancers charges APC — confirm institutional waiver or payment arrangement before submitting
- [ ] **Author ORCID iDs**: MDPI requires ORCID for all authors — collect ORCID iDs from all co-authors
- [ ] **Supplementary materials URL** (manuscript line 664): `https://www.mdpi.com/article/XXX/s1` — this URL is assigned after acceptance. Delete this line or mark as TBD for initial submission.
- [ ] **Section**: Confirm "Gastrointestinal Cancers" is the correct section for this article
- [ ] **Word count**: MDPI Cancers has no strict word limit for Research Articles but recommends ≤8,000 words — verify current count

---

## STEP 4 — Recompile PDF
```bash
cd C15_esophageal/manuscript
xelatex c15_manuscript.tex
xelatex c15_manuscript.tex   # second pass for cross-references
```

---

## STEP 5 — Figure resolution check
Current figures are at 150 dpi. MDPI requires **300 dpi** for raster figures.

Figures referenced in manuscript:
- `results/02_descriptive/fig_age_sex.png` + `fig_histology_stage.png` + `fig_incidence_trend.png`
- `results/03_survival/km_stage.png` + `km_histology.png` + `km_surgery.png`
- `results/06_chemo_surgery/cox_treatment_forest.png` + `km_ccrt.png`
- `results/04_deep_learning/umap_kmeans_k3.png` + `km_clusters_k3.png` + `deepsurv_feature_importance.png`

Regenerate at 300 dpi: add `dpi=300` to all `savefig()` calls in scripts 02, 03, 04, 06,
then rerun: `bash analysis/run_all.sh`

---

## STEP 6 — Highlights (not required by MDPI; skip)
MDPI Cancers does not require a separate Highlights file (unlike Elsevier).

---

## STEP 7 — MDPI submission portal
URL: https://susy.mdpi.com/user/manuscripts/upload

Files to upload:
- [ ] Manuscript PDF
- [ ] Manuscript .tex source
- [ ] Cover letter
- [ ] Each figure as separate file (TIFF or PNG ≥ 300 dpi)
- [ ] Supplementary figure S1 (KM bootstrap — referenced at line 664)

---

## STEP 8 — Final checks
- [ ] All author ORCID iDs collected
- [ ] APC confirmed / waiver arranged
- [ ] "Single-center" vs "population-based" language resolved in cover letter
- [ ] "Largest cohort" claim removed or qualified
- [ ] Supplementary URL placeholder removed for initial submission
- [ ] Received/Accepted/Published date placeholders deleted (MDPI fills post-acceptance)
- [ ] Publisher's Note placeholder deleted (MDPI fills post-acceptance)
- [ ] Conflict of interest: "None declared" ✓
- [ ] Funding: "No external funding" ✓ — confirm with all authors

---

## Summary — what only you can fill

| Item | Location | Status |
|------|----------|--------|
| Author 1–3 full names | tex lines 59–61 | ⬜ |
| Corresponding author email | tex line 69 | ⬜ |
| Corresponding author phone | tex line 70 | ⬜ |
| Department + institution + city | tex lines 66–67 | ⬜ |
| Second affiliation (or delete) | tex line 68 | ⬜ |
| IRB institution + number | tex lines 286, 682–683 | ⬜ |
| CRediT roles confirmed | tex lines 670–673 | ⬜ |
| ORCID iDs × 3 | MDPI portal | ⬜ |
| APC / waiver | MDPI portal | ⬜ |
| Cover letter header + sign-off | cover letter lines 1–6, 67–70 | ⬜ |
| IRB + data release agreement | cover letter lines 54–56 | ⬜ |
| Suggested reviewers × 3 | cover letter lines 76–78 | ⬜ |
| Figures at 300 dpi | run_all.sh | ⬜ |
| Supplementary URL removed | tex line 664 | ⬜ |
| Date placeholders deleted | tex lines 74–75, 78 | ⬜ |
