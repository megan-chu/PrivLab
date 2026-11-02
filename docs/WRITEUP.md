# PrivLab: write-up (draft)

Numbers marked PLACEHOLDER come from the final test run (Freeze 7).

## The problem
People who work with reports spend hours retyping data from them: tumour size, stage, account numbers, dates. The
reports are full of personal details, so they cannot simply be shared or given to an AI to read. The result is slow
manual work, or AI used without proper care.

## What PrivLab does
PrivLab removes the personal details on the user's own device, then fills in the fields the user chooses, each value
traced back to the words it came from. It runs as a local viewer and as an MCP server, so it plugs into the AI
assistants people already use.

1. **Review identifiers (on the device).** Patterns, a small local name model and a hold rule find names, IDs,
   phones, addresses, card and bank details. The user confirms what to mask. Nothing has been sent.
2. **De-identify.** Each detail becomes a tag such as [PERSON-1]. Only this masked text goes to Muse Spark Open, once,
   as a safety check.
3. **Choose fields.** The user describes a field in their own words; Muse Spark 1.3 drafts a precise definition for
   them to approve. No report text is sent for this.
4. **Extract.** Muse Spark reads the masked text and proposes each value; code checks that every value is really in
   its quote. The user confirms or corrects.
5. **Export.** A CSV with one row per report, the masked text, or both, built on the device.

It learns from the user: an identifier hidden once is caught on the device next time, and confirmed or corrected
values become examples for the next reports. A field is marked Ready only when, with 95% confidence, at most 5% of
its verified values are wrong.

## How we use Meta's models
- **Muse Spark Open (open weights, on Baseten):** the de-identification safety check. Because the weights are open, a
  hospital can host it itself by changing two settings, so nothing reaches an outside API during de-identification.
- **Muse Spark 1.3 (Meta Model API):** field definitions and extraction, with strict JSON output, one cached call per
  report.

## Results (synthetic data)
- De-identification on the device, before any API call: dev and extra reports 100% of 1,876 identifiers; hand-written
  hard cases 98.5% of 130; frozen uncued-names benchmark 94.6% of 56; 0 false alarms on every set.
- Held-out test reports: de-identification PLACEHOLDER; field accuracy (F1) PLACEHOLDER against the user's confirmations.
- Indicative comparison: LLM-AIx reports 87–90% on similar fields without learning (different reports).

## Privacy by design
Only one module can call an API, and it accepts report text only as a sealed, masked object. Every outgoing prompt is
re-scanned and refused if an identifier pattern is left. Details and residual risk: `docs/PRIVACY_NOTE.md`.

## What's next
Host the safety check in-house; OCR for scanned reports; more languages and name forms.
