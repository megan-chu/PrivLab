# PrivLab
Healthcare professionals spend hours reading reports and retyping data. Reports are rich in information but full of patients’ personal details, so they can’t simply share them or use AI to analyse them. PrivLab removes those details locally, then extracts the data they need, as an MCP server that plugs into the tools they already use.

## Input

PrivLab reads plain text (a `.txt` file, or pasted text) and PDFs with selectable text. PDF text is
extracted on the laptop; a page without text is skipped with a warning. Scanned PDFs, images and other
formats are refused with a message saying they are not supported: this version has no OCR, and report
content is never sent to a cloud API to be read. Accuracy has been measured on pathology reports only.

## Privacy: what reaches the cloud, and residual risk

PrivLab de-identifies in two layers. **Presidio pattern recognisers** run on the laptop and mask
structured identifiers (NRIC/FIN, NHS and SSN-format numbers, MRN, accession numbers, phones, emails,
dates, addresses, labelled names and labelled IDs in any report type), and a small spaCy name model
(`en_core_web_sm`, on the laptop, ~6 ms per report) fills in names without a cue. The model only adds
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
| **Human review before anything leaves the laptop.** The user reviews and corrects the local highlights first; Muse Spark only sees text a person has approved, and anything new it flags gets a second, short review (pre-ticked "hide"). | Built |
| **Hold rule.** If masked text still contains something that looks like a name (capitalised words not in the report vocabulary, lower-case "x binte y" names), PrivLab pauses and asks before calling the API. | Built |
| **Two exports, kept apart.** The registry export (CSV) holds a random case id, report id, field values and evidence positions: no report text, so a missed name does not travel onward with the data. The masked report text (one .txt, or a .zip of every finished report) is a separate download the user chooses; a report whose final scan still finds an identifier pattern is withheld from both. | Built |
| **Model qualification.** The name check uses a model only after it passes a frozen qualification set under its exact settings (provider, model, effort, prompt). Muse Spark Open and Muse Spark 1.3 both pass (11 of 12, no false alarms); a changed setting needs a new run, so swapping models cannot silently lower privacy. | Built |
| **Measured exposure.** Evaluation reports the share of planted identifiers that reach the API (missed locally) and the share left after both layers. | Built |
| **Learns from corrections without leaking.** Confirmed values become examples for later prompts; an example that still contains an identifier pattern is never stored. | Built |
| **Open-weights second layer, hostable in-house.** The name check runs on Muse Spark Open. For production, the hospital hosts the same weights on its own servers and changes two settings (`BASETEN_BASE_URL`, `BASETEN_MODEL`), so nothing reaches an outside API. | Open-weights model: built · in-hospital hosting: roadmap (config only) |
| **Governance.** Real use needs hospital approval, a data-processing agreement with the API provider and a review of its data terms. | Out of scope for the hackathon |

**Current results (synthetic data, development sets).** Locally (patterns + hold rule, no API call), PrivLab
catches 100% of the 1,876 planted identifiers in the dev and extra reports, with no false alarms. That is
partly an artefact of the data: every planted name follows a cue such as "Patient:" or "Dr". On hand-written
hard cases (names and places without cues) Presidio alone caught 8%; with the hold rule and the Muse Spark name check,
0% remain. The hard cases were also used to improve the rules, so these are development numbers; the
held-out test reports give the final figures.
The name check runs on Muse Spark Open so that de-identification can be hosted locally. In an early
test, before the hold rule caught them locally, it missed 2 of 49 hard names that Muse Spark 1.3 caught;
Muse Spark 1.3 stays available as a backup, and the held-out test run reports the final figures.
