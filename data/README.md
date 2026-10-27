# PrivLab synthetic dataset

200 real cancer pathology reports from TCGA-Reports, with **fake** identifiers planted in them.
No real patient data: TCGA text had its identifiers removed by the dataset authors, and every
name, ID, date, phone, address and email here was randomly generated (seed 42).

## Files

| Path | What it is |
| --- | --- |
| `reports/PPxxxx.txt` | Report text with planted identifiers (what your tools receive) |
| `truth/phi/PPxxxx.json` | De-identification answer key: every planted span with `type`, `text`, `start`, `end` |
| `manifest.csv` | report_id, TCGA ids, cancer_type, split, country format, number of identifiers |
| `pdf/` | 5 hospital-style PDFs for the demo video (one per cancer type, from the `extra` split) |
| `raw/tcga_starter_200.csv` | Source TCGA text and cancer-type labels |

## Splits (stratified, 5 cancer types: BRCA, COAD, KIRC, LUAD, PRAD)

- `dev`: 20 reports (4 per type). Look at these while building.
- `test`: 50 reports (10 per type). Do not look until final scoring.
- `extra`: 130 reports. Use for demos, prompt examples and labelling practice.

## Identifier types in the answer key

`NAME`, `ID` (NRIC / NHS / SSN-format, checksum-valid, not real), `DOB`, `DATE`, `MRN`, `ACCESSION`,
`DOCTOR`, `HOSPITAL`, `PHONE`, `ADDRESS`, `EMAIL`, `ACCOUNT`.

Where they appear: header block, a barcode-style line (~40%), blank TCGA field labels filled in place,
a free-text comment (~70%: a relative's name, a phone call, an email), the signature line, and a footer
repeating the patient name (~50%).

Cite: Kefeli J, Tatonetti N. TCGA-Reports: A machine-readable pathology report resource for
benchmarking text-based AI models. *Patterns* 2024;5(3):100933.
