# Submission Checklist — Co-existing Cancer Patterns
## Target: Cancer Epidemiology (Elsevier) — Original Research Article

---

## STEP 1 — Fill placeholders in manuscript
**File:** `manuscript/coexist_cancers_manuscript.tex`

| Line | Placeholder | Fill with |
|------|-------------|-----------|
| 42 | `[Author 1 Name]` | First author full name |
| 43 | `[email@institution.edu.tw]` | Corresponding author email |
| 44 | `[Author 2 Name]` | Second author full name |
| 45 | `[Author 3 Name]` | Third author full name |
| 47 | `[Department of Oncology / Epidemiology]` | Actual department name |
| 47 | `[Institution Name]` | Hospital / university name |
| 48 | `[City]` | City name (e.g. Taichung) |
| 49 | `[Second Affiliation, if applicable]` | Delete line if not needed; otherwise fill |
| 51 | `+886-X-XXXX-XXXX` | Corresponding author phone |
| 269 | `[Institution]` | IRB-issuing institution (appears twice) |
| 269 | `[IRB-XXXX-XXXX]` | IRB approval number (appears twice: lines 269, 521-522) |
| 527 | `[Author 1]` | First author name (contributions) |
| 528 | `[Author 2]` | Second author name (contributions) |
| 529 | `[Author 3]` | Third author name (contributions) |

> Note: Author contributions on lines 527–529 are pre-written as:
> - Author 1: Conceptualisation, Methodology, Software, Formal analysis, Writing — original draft
> - Author 2: Data curation, Validation
> - Author 3: Supervision, Writing — review and editing
>
> Adjust roles if they do not match actual contributions.

---

## STEP 2 — Fill placeholders in cover letter
**File:** `results/cover_letter_CancerEpidemiology.md`

| Line | Placeholder | Fill with |
|------|-------------|-----------|
| 1 | `[Corresponding Author Name, MD/PhD]` | Name + degree |
| 2 | `[Department / Division]` | Department |
| 3 | `[Institution Name]` | Hospital / university |
| 4 | `[Address Line 1]` | Street address |
| 5 | `[City, Country, Postal Code]` | e.g. Taichung, Taiwan 404 |
| 6 | `[email@institution.edu.tw]` | Email |
| 6 | `[+886-X-XXXX-XXXX]` | Phone |
| 50–51 | `[Institution]`, `[IRB-XXXX-XXXX]` | Same as manuscript |
| 61 | `[Corresponding Author Name, MD/PhD]` | Repeat name + degree |
| 62 | `[Title, Department, Institution]` | e.g. Assistant Professor, Dept of Oncology, CMUH |
| 63 | `[Email]` | Email |
| 68 | `[Name], [Institution] — multiple primary cancers / cancer registry epidemiology` | Suggested reviewer 1 |
| 69 | `[Name], [Institution] — head and neck / esophageal field cancerization` | Suggested reviewer 2 |
| 70 | `[Name], [Institution] — machine learning in cancer epidemiology` | Suggested reviewer 3 |

> If you prefer not to suggest reviewers, delete lines 68–70 and the header "Suggested reviewers".

---

## STEP 3 — Recompile PDF
```bash
cd coexist_cancers/manuscript
xelatex coexist_cancers_manuscript.tex
xelatex coexist_cancers_manuscript.tex   # second pass for cross-references
```
Check: all figures load, no missing references, page count ≤ journal limit.

---

## STEP 4 — Verify against Cancer Epidemiology author guidelines
URL: https://www.sciencedirect.com/journal/cancer-epidemiology/publish/guide-for-authors

Key limits to check:
- [ ] Abstract: ≤ 300 words (currently ~252 — OK)
- [ ] Main text word limit (Original Research): typically 4,000–5,000 words (excluding abstract/refs)
- [ ] Figures: 7 total — confirm each is ≥ 300 dpi or vector (current figures are PNG from matplotlib at 150 dpi — **may need regeneration**)
- [ ] Tables: 3 total — confirm fit within journal column width
- [ ] References: 17 — confirm format matches journal style (numbered, Vancouver)
- [ ] Structured abstract: Background / Methods / Results / Conclusions — confirm all four headings present

---

## STEP 5 — Figure resolution check
Current figures were generated at 150 dpi (matplotlib default). Cancer Epidemiology requires **300 dpi minimum** for raster figures.

Figures used in manuscript (check `\graphicspath{{../results/}}`):
- `01_matrix/cancer_prevalence.png`
- `01_matrix/cooccurrence_heatmap.png`
- `02_associations/lift_heatmap.png`
- `02_associations/top_associations_lift.png`
- `03_nmf/nmf_programs_multiprimary.png`
- `04_clustering/umap_kmeans_k3.png`
- `05_sir_trajectories/sir_forest.png`
- `05_sir_trajectories/trajectory_graph.png`

To regenerate all figures at 300 dpi, add `dpi=300` to every `fig.savefig(...)` call in:
- `analysis/01_build_patient_matrix.py`
- `analysis/02_association_rules.py`
- `analysis/03_nmf_patterns.py`
- `analysis/04_autoencoder_clustering.py`
- `analysis/07_sir_trajectories.py`

Then rerun: `bash analysis/run_all.sh`

---

## STEP 6 — Assemble submission package
Cancer Epidemiology uses **Editorial Manager** (Elsevier submission system).

Files to upload:
- [ ] Manuscript PDF (compiled from filled .tex)
- [ ] Manuscript .tex source file
- [ ] Cover letter (paste text or upload .docx/.pdf)
- [ ] Each figure as a separate file (TIFF or high-res PNG, ≥ 300 dpi)
- [ ] Highlights file: 3–5 bullet points, ≤ 85 characters each (see below)

---

## STEP 7 — Write Highlights (required by Elsevier)
Elsevier requires a "Highlights" file: 3–5 bullet points, each ≤ 85 characters.

Draft (edit as needed):
```
• Three independent ML methods converge on the same cancer co-occurrence axes
• Aerodigestive SCC field drives 13% third-primary rate (cohort-wide: 5.2%)
• Esophagus→larynx SIR = 12.2: highest second-primary risk quantified
• Aerodigestive pairs arise synchronously (60% within 6 months)
• Findings support pan-aerodigestive surveillance at first diagnosis
```
Check each is ≤ 85 characters. Save as `manuscript/highlights.txt`.

---

## STEP 8 — Final checks before submit
- [ ] All author names spelled correctly and in agreed order
- [ ] No track changes or comments left in the document
- [ ] Conflict of interest statement: "The authors declare no competing interests."
- [ ] Funding statement: "No external funding was received for this study."
- [ ] Data availability: "Data are available from the Taiwan Cancer Registry subject to the registry's data-access governance."
- [ ] Confirm corresponding author will handle all editorial correspondence
- [ ] Save a dated copy of the submitted PDF in `results/` for records

---

## Summary — what only you can fill

| Item | Where | Status |
|------|-------|--------|
| Author 1 full name | tex line 42, cover letter | ⬜ |
| Author 2 full name | tex line 44, cover letter | ⬜ |
| Author 3 full name | tex line 45, cover letter | ⬜ |
| Corresponding author email | tex line 43 | ⬜ |
| Corresponding author phone | tex line 51 | ⬜ |
| Department + institution | tex lines 47–48 | ⬜ |
| Second affiliation (or delete) | tex line 49 | ⬜ |
| IRB institution + number | tex lines 269, 521–522 | ⬜ |
| Author contributions (roles) | tex lines 527–529 | ⬜ |
| Suggested reviewers × 3 | cover letter lines 68–70 | ⬜ |
| Cover letter header block | cover letter lines 1–6 | ⬜ |
| Cover letter sign-off block | cover letter lines 61–63 | ⬜ |
| Highlights file | new file: manuscript/highlights.txt | ⬜ |
| Figures regenerated at 300 dpi | run_all.sh | ⬜ |
