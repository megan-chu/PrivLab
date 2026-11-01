# PrivLab
Healthcare professionals spend hours reading reports and retyping data. Reports are rich in information but full of patients’ personal details, so they can’t simply share them or use AI to analyse them. PrivLab removes those details locally, then extracts the data they need, as an MCP server that plugs into the tools they already use.

![How PrivLab works](docs/architecture.png)

## Run it (Windows, Python 3.11+)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env        # add your META_API_KEY and BASETEN_API_KEY; generate the two local keys as the file says
.venv\Scripts\python -m pytest -q                       # no API calls: tests use a fake model
.venv\Scripts\python -m viewer.server                   # open http://127.0.0.1:8765
```
MCP (Muse Code, Claude Desktop or any MCP client): a stdio server with command `.venv\Scripts\python.exe` and argument
`mcp_server\server.py` (full paths). De-identification scores, no API calls: `.venv\Scripts\python -m eval.score_deid --split dev`.

## Input

PrivLab reads plain text (a `.txt` file, or pasted text) and PDFs with selectable text. PDF text is
extracted on the laptop; a page without text is skipped with a warning. Scanned PDFs, images and other
formats are refused with a message saying they are not supported: this version has no OCR, and report
content is never sent to a cloud API to be read. Accuracy has been measured on pathology reports only.

## Privacy: what reaches the cloud, and residual risk

PrivLab de-identifies in two layers. **Presidio pattern recognisers** run on the laptop and mask
structured identifiers (NRIC/FIN, NHS and SSN-format numbers, MRN, accession numbers, CLIA numbers, phones, emails,
dates, addresses, company and bank names, card numbers and IBANs (checksum-verified), labelled names and labelled IDs in any report type), and a small spaCy name model
(`en_core_web_sm`, on the laptop, ~6 ms per report) fills in names without a cue. A hold rule then asks about leftover
words that look like a person or place, and corrected names are learned for next time. The model only adds
names: it never overrides a pattern, and the final leftover scan stays rules-only. **Muse Spark Open** (Meta's open-weights model, hosted on Baseten) then looks for names and
other free-text details in the already-masked text. Because it is open weights, the whole de-identification
step can later run inside a hospital: only two settings change, no code. Extraction only ever receives de-identified text.

The data in this repo is synthetic: real TCGA pathology text with randomly generated identifiers.
For real hospital data, the two layers alone are not enough, because an identifier can slip past them:

| Missed by | What happens | Risk |
|---|---|---|
| Presidio only | Muse Spark sees it and usually catches it | The identifier has reached an external API |
| Presidio and Muse Spark | It stays in the de-identified text | It reaches the extraction step and stays in the cleaned report |

### How PrivLab reduces that risk

| Measure | Status |
|---|---|
| **Code-level guard.** Only one module can call the API. Report text reaches it only as a sealed, masked or de-identified object (never a plain string); the one plain-text prompt is a field definition the user types. Every outgoing prompt, including learned examples, is re-scanned, and sending is refused if an identifier pattern is still present. | Built |
| **Human review before anything leaves the laptop.** The user reviews and corrects the local highlights first; Muse Spark only sees text a person has approved, and anything new it flags gets a second, short review (pre-ticked "hide"). Corrections are staged per report and saved when it finishes; Back undoes them. | Built |
| **Hold rule.** If masked text still holds a person or place the patterns missed, PrivLab pauses and asks before calling the API — but only on evidence: a listed name, a person cue ("signed by", "Dr", ", M.D."), a place cue ("lives in"), an institution word, or a person tag beside it. Drug names, genes, biomarkers, units and lab labels are not asked about. | Built |
| **Two exports, kept apart.** The data export (CSV) has one row per report: a random case id, report id, review time and one column per field, values as written: no report text, so a missed name does not travel onward with the data. The masked report text (one .txt, or a .zip of every finished report) is a separate download the user chooses; **Export everything** puts both in one .zip with a data dictionary (each column's type and the field's definition and version). Only the reports in the current list are exported; a list field's items are joined by ";" in its column, with one row per item in lists/; every download first says what it holds and what it left out, and why. A report whose final scan still finds an identifier pattern is withheld from both. | Built |
| **Model qualification.** The name check uses a model only after it passes a frozen qualification set under its exact settings (provider, model, effort, prompt). Muse Spark Open and Muse Spark 1.3 both pass (11 of 12, no false alarms); a changed setting needs a new run, so swapping models cannot silently lower privacy. | Built |
| **Measured exposure.** Evaluation reports the share of planted identifiers that reach the API (missed locally) and the share left after both layers. | Built |
| **Learns from corrections without leaking.** Confirmed values become examples for later prompts; an example that still contains an identifier pattern is never stored. Identifiers you hide are caught on the laptop on the next report ("Learned from report_1"). Everything learned is listed under Fields → Learned, where it can be removed; values saved by Extract all, and reset values, never teach. | Built |
| **Accuracy you can audit.** The person who defines a field validates it. Each field has a precise definition and a version (a new meaning restarts its score); about 1 in 10 values is a blind check (PrivLab's answer hidden until you answer); a field is Ready only when, with 95% confidence, at most 5% of its verified values are wrong (59 verified values with no error). Every check is in the field's history. | Built |
| **Open-weights second layer, hostable in-house.** The name check runs on Muse Spark Open. For production, the hospital hosts the same weights on its own servers and changes two settings (`BASETEN_BASE_URL`, `BASETEN_MODEL`), so nothing reaches an outside API. | Open-weights model: built · in-hospital hosting: roadmap (config only) |
| **Chat through MCP.** The tools take file paths, never report text, and answer with ids, types and masked context; the de-identified text only after the review. So the assistant's model sees the same masked text Muse Spark sees. A name the user types into the chat to hide it does reach the assistant's provider, so the tools point to the local viewer for that. A field definition you write is sent as written. | Built |
| **Governance.** Real use needs hospital approval, a data-processing agreement with the API provider and a review of its data terms. | Out of scope for the hackathon |

**Current results (synthetic data, development sets).** Locally (patterns + hold rule, no API call), PrivLab
catches 100% of the 1,876 planted identifiers in the dev and extra reports, with no false alarms. That is
partly an artefact of the data: every planted name follows a cue such as "Patient:" or "Dr". On hand-written
hard cases (130 identifiers: names, places, addresses, phones, IDs, card and bank details and mailing blocks
without cues) the local patterns and name model alone catch 93.1% (Presidio patterns alone caught 8% at the start);
with the hold rule, 97.7% are caught before any API call, with 0 false alarms; the rest reach the second check.
The hard cases were also used to improve the rules (including a 15-name development set written for that), so
these are development numbers; the held-out test reports give the final figures.
**Uncued-names benchmark** (56 invented names in 48 short narratives with no cue such as "Patient:" or
"Dr", Chinese, Malay, Indian and Western names, a quarter in lower case; generated once, frozen by hash,
never used for tuning): the local patterns and name model catch 83.9%; with the hold rule, 91.1% are caught
before any API call, with 0 false alarms (was 61% and 82% on 30 Oct). On a US genomic sample report, hold questions
fell from 23 to 0 and identifiers reaching the cloud from 4 to 0. With the Muse Spark Open name check (30 Oct, older
rules), 95% (53 of 56); 14% of names reached the API unmasked then, which is the exposure a hospital would host the
model locally to remove.
The name check runs on Muse Spark Open so that de-identification can be hosted locally. In an early
test, before the hold rule caught them locally, it missed 2 of 49 hard names that Muse Spark 1.3 caught;
Muse Spark 1.3 stays available as a backup, and the held-out test run reports the final figures.
