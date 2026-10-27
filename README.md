# PrivLab
Healthcare professionals spend hours reading reports and retyping data. Reports are rich in information but full of patients’ personal details, so they can’t simply share them or use AI to analyse them. PrivLab removes those details locally, then extracts the data they need, as an MCP server that plugs into the tools they already use.

## Privacy: what reaches the cloud, and residual risk

PrivLab de-identifies in two layers. **Presidio pattern recognisers** run on the laptop and mask
structured identifiers (NRIC/FIN, NHS and SSN-format numbers, MRN, accession numbers, phones, emails,
dates, addresses and labelled names). **Muse Spark 1.3** (Meta Model API) then looks for names and
other free-text details in the already-masked text. Extraction only ever receives de-identified text.

The data in this repo is synthetic: real TCGA pathology text with randomly generated identifiers.
For real hospital data, the two layers alone are not enough, because an identifier can slip past them:

| Missed by | What happens | Risk |
|---|---|---|
| Presidio only | Muse Spark sees it and usually catches it | The identifier has reached an external API |
| Presidio and Muse Spark | It stays in the de-identified text | It reaches the extraction step and stays in the cleaned report |

### How PrivLab reduces that risk

| Measure | Status |
|---|---|
| **Code-level guard.** Only one module can call the API, it accepts only masked or de-identified objects (never raw text), and it re-scans every outgoing text, refusing to send if an identifier pattern is still present. | Built |
| **Measured exposure.** Evaluation reports the share of planted identifiers that reach Muse Spark (missed by Presidio) and the share left after both layers. | Presidio layer measured; Muse Spark layer planned |
| **Human review before anything leaves the laptop.** The user reviews and corrects the local highlights first; Muse Spark only sees text a person has approved, and anything new it flags gets a second, short review. | Planned |
| **Hold rule.** If masked text still contains something that looks like a name (capitalised words not in a medical or English word list), PrivLab pauses and asks before calling the API. | Planned |
| **Share structured output only.** Registry exports contain the extracted fields and a case tag, never the free text, so a missed name does not travel onward. | Planned |
| **Self-hosted second layer.** For production, run the name check on Muse Glimmer (Meta open weights) on the hospital's own servers, so nothing reaches an outside API. | Roadmap |
| **Governance.** Real use needs hospital approval, a data-processing agreement with the API provider and a review of its data terms. | Out of scope for the hackathon |

**Current results (synthetic data).** Presidio alone covers 100% of the planted identifiers on the
dev and extra splits, so none of them would reach Muse Spark. This is partly an artefact of the data:
every planted name follows a cue such as "Patient:" or "Dr". Names written in free text without a cue
are missed by patterns alone; that is the job of the Muse Spark layer and the human review step.
