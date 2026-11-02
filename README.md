# PrivLab

**Reports in. Structured data out. Personal details stay on your device.**

![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![Runs locally](https://img.shields.io/badge/de--identification-on%20device-2e7d32) ![MCP](https://img.shields.io/badge/MCP-server-6a4fb3) ![Muse Spark 1.3](https://img.shields.io/badge/Meta-Muse%20Spark%201.3-0866ff) ![Muse Spark Open](https://img.shields.io/badge/Meta-Muse%20Spark%20Open%20%28open%20weights%29-0866ff)

People who work with reports (pathology, lab results, claims, statements) spend hours reading them and retyping
the same values into spreadsheets. The reports are full of personal details, so they can't simply be shared, or
handed to an AI to read.

PrivLab fixes that. It removes the personal details **on your own device**, then fills in the fields you choose,
with every value traced back to the words it came from. It runs as a simple web page on your computer, or as an
MCP server inside the AI assistant you already use.

Built solo for the **Meta Global AI Developer Hackathon** (Track 1, Indie developer), with two Meta models:
**Muse Spark 1.3** through the Meta Model API, and open-weights **Muse Spark Open**.

## At a glance

| | |
|---|---|
| 🌍 **Impact** | Cancer registrars, clinical researchers, claims and finance teams retype values from sensitive reports by hand, because privacy rules stop them using AI. PrivLab lets them: personal details never leave the device, and every value comes with its source for a quick check. |
| 🦙 **Use of Meta AI** | Each Meta model has its own job. Open-weights **Muse Spark Open** double-checks the masking and can be hosted in-house (Muse Spark 1.3 is its qualified backup). **Muse Spark 1.3** turns a field described in plain words into a precise definition, then extracts every field in one checked JSON call, steered by examples learned from your corrections. A model is used only after passing a privacy qualification test. As an **MCP server**, PrivLab gives assistants such as Muse Code private access to reports. |
| 🛠️ **Technical depth** | On-device detection of personal details (patterns, checksums, a local name model, address parsing), a privacy guard built into the code, values verified against their quotes, learning on device, and a field marked Ready only by an exact 95% confidence bound on its error rate. 260 automated tests. |
| 🎬 **Demo** | Add a report, then five steps from review to export, end to end on built-in samples; plus the MCP server. 100% of 1,876 personal details caught on device in the development reports, 0 false alarms; held-out test scores follow the final run (see [Results](#results)). |

## Contents
- [At a glance](#at-a-glance)
- [What makes it different](#what-makes-it-different)
- [How it works](#how-it-works)
- [Getting started](#getting-started)
- [Choose how to use PrivLab](#choose-how-to-use-privlab)
- [Using PrivLab, step by step](#using-privlab-step-by-step)
- [Use it from an AI assistant (MCP)](#use-it-from-an-ai-assistant-mcp)
- [Privacy at a glance](#privacy-at-a-glance)
- [Results](#results)
- [Running the tests](#running-the-tests)
- [What's next](#whats-next)

## What makes it different

- 🔒 **Private first.** Names, ID numbers, phones, addresses, card and bank details are found and masked on your
  device. You confirm them before anything is sent, and the AI only ever sees masked text.
- 🎯 **Your fields, your words.** Describe a field the way you would to a colleague ("tumour size", "account
  balance"). PrivLab drafts a precise definition for you to approve.
- 🔎 **Every value has a source.** Each extracted value is highlighted in the report with its quote, and checked
  in code that it really appears there.
- 📈 **It learns from you.** A name you hide once is caught on your device the next time. Values you confirm or
  correct guide the next reports.
- ✅ **Accuracy you can trust.** A field is marked *Ready* only when your own checks show, with 95% confidence,
  that at most 5% of its values are wrong. About 1 in 10 values is a blind check, where PrivLab hides its answer
  until you give yours.
- 📦 **One report or a whole batch.** Review each report yourself, or extract a batch in one go and only check
  what needs you.

## How it works

```mermaid
flowchart LR
    A["📄 Add reports<br/>.txt, pasted text<br/>or text PDF"] --> B["1 · Review identifiers<br/>🔒 on your device<br/>you confirm what to mask"]
    B --> C["2 · De-identify<br/>details become tags<br/>like [PERSON-1]"]
    C --> D["3 · Choose fields<br/>describe them in<br/>your own words"]
    D --> E["4 · Extract<br/>AI reads masked text;<br/>you confirm each value"]
    E --> F["5 · Export<br/>CSV · masked text<br/>· or both in a .zip"]
    E -. "your corrections" .-> L(["📈 PrivLab learns"])
    L -.-> B
    L -.-> E
```

| Step | What happens | What leaves your device |
|---|---|---|
| **1 · Review identifiers** | PrivLab highlights every personal detail it finds. Click to keep a word, select text to hide more. | Nothing |
| **2 · De-identify** | Each detail becomes a tag, so `Sarah Jenkins (MRN 849204)` reads `[PERSON-1] (MRN [MRN-1])`. | The masked text, once, as a safety check |
| **3 · Choose fields** | Tick the fields you need, or add one in plain words. You approve its definition. | The field name and your description only |
| **4 · Extract** | The AI proposes each value with its quote. You confirm, correct, or mark it "not stated". | Masked text only |
| **5 · Export** | Download the data as a spreadsheet-ready CSV, the masked text, or both. | Nothing: files are built on your device |

![PrivLab architecture](docs/architecture.png)

## Getting started

You need Windows with Python 3.11 or newer.

```powershell
git clone <this repository's URL> PrivLab
cd PrivLab
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and add your Meta Model API key and Baseten key. The file explains how to create the two local
security keys (one command each).

Start PrivLab: **double-click `PrivLab.cmd`** in the PrivLab folder. It opens PrivLab in your browser
(http://127.0.0.1:8765); close its window to stop. Using PrivLab from an AI assistant instead? You don't need this:
PrivLab starts its page by itself when the assistant connects, and the assistant gives you the link.

From a terminal, the same: `.venv\Scripts\python -m viewer.server --open`.

## Choose how to use PrivLab

| You want to… | Use | How |
|---|---|---|
| Check a few reports carefully, by eye | **The browser** | Double-click `PrivLab.cmd`, add reports, follow the five steps |
| Process many reports, or combine with other work | **Your AI assistant** (e.g. Muse Code) | Say *"Fill my fields for C:/reports/case12.txt with PrivLab"* |
| Both: talk for speed, look when it matters | **Both together** | Start in the assistant; each report opens in the browser by itself, and the browser lists every report you added in the chat |

Same reports, same fields, same learning either way, and the same rule: originals stay on your computer, nothing is
sent until you confirm, and any AI only sees masked text.

## Using PrivLab, step by step

Steps 1–2 get a report in; steps 3–7 are the five steps of the workflow above.

1. **Add a report.** Drop a `.txt` or text PDF on the page, paste text with Ctrl+V, or click a sample
   (Breast, Colon, Kidney, Lung, Prostate).
2. **Choose your task** at the top: *De-identify only*, or *De-identify + extract*.
3. **Review identifiers.** Check the highlights. Click a highlight to keep that word visible; select any text to hide
   it. Then click **Confirm N identifiers for masking**.
4. **De-identify.** Read the masked text, then **Confirm and choose fields** (or download the masked text and stop here).
5. **Choose fields.** Tick the fields you want. To add one, click **+ Add a field**, describe it, give an example value,
   and approve the definition PrivLab drafts.
6. **Extract.** Each value appears in the report in its own colour, with its quote. Confirm it, select the right text
   to correct it, or mark it *not stated*. Then **Finish and export this report**.
7. **Export.** Choose **Extracted data (.csv)**, **Masked reports (.zip)**, or **Everything (.zip)**, which adds a data
   dictionary. Before each download, PrivLab tells you what is in the file and what was left out.

**Working on a batch?** Add all your reports, review each one's identifiers, choose the fields once, then click
**Extract all**. Values PrivLab verified are saved for you; anything that needs a human is listed for you to check.

**See what it learned:** **Fields → Learned** lists everything PrivLab has learned from you, and you can remove any of it.

## Use it from an AI assistant (MCP)

PrivLab is also an MCP server, so Muse Code or any MCP-compatible assistant can use it. Add a stdio server:

- **Command:** the full path to `.venv\Scripts\python.exe`
- **Argument:** the full path to `mcp_server\server.py`

Then ask, for example: *"Fill my fields for C:/reports/report_1.txt with PrivLab."* The assistant only ever sees
tags and masked text. It gives you a link to open the same report in the PrivLab page to review the real text.

PrivLab gives the assistant **five tools, one per step**: `privlab_deidentify`, `privlab_review`, `privlab_fields`,
`privlab_extract` (all your answers in one call) and `privlab_export`, plus four ready-made prompts ("Fill my fields",
"De-identify a batch", "Set up my fields", "Open the PrivLab viewer"). They follow common MCP design practice:
outcome-based tools, flat arguments, "Use when…" instructions, service-prefixed names and paged results.

## Privacy at a glance

- **On your device:** reading the file, finding personal details, masking, the vault that links tags back to
  originals (encrypted), and building every export.
- **Sent, only after you confirm:** masked text, to Meta's **Muse Spark Open** (a safety check for anything missed)
  and **Muse Spark 1.3** (field definitions and extraction).
- **Built into the code, not left to good intentions:** only one part of PrivLab can contact an AI, and it refuses
  anything that isn't masked. Every message is scanned again before sending.
- **Ready for in-house hosting:** the safety check uses open-weights Muse Spark Open, so an organisation can run it
  on its own servers by changing two settings.

| Meta model | Where | Job in PrivLab | Why this model |
|---|---|---|---|
| **Muse Spark Open** | open weights, hosted on Baseten | Second check: finds names the on-device rules missed, in already-masked text | Open weights: if an organisation hosts it itself, nothing leaves during de-identification |
| **Muse Spark 1.3** | Meta Model API | Drafts a precise definition from your words; extracts every field in one call per report; backup for the second check | Strong reading of long reports; strict JSON output that code can check |

All data in this repository is synthetic: real public pathology text (TCGA) with invented personal details.

## Results

Measured on synthetic data, on the device, **before any AI call**. Scored the way clinical de-identification benchmarks
are: each personal detail is fully hidden, partly hidden (part still shows) or missed, and every word is checked
(precision: hidden words that really were personal; recall: personal words that were hidden).

| Test set | Personal details | Fully hidden | Partly / missed | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Pathology reports (dev) | 253 | **100%** | 0 / 0 | 100% | 100% | **1.00** |
| Pathology reports (extra) | 1,623 | **100%** | 0 / 0 | 100% | 100% | **1.00** |
| Tricky formats (red-team) | 30 | **100%** | 0 / 0 | 98.8% | 100% | **0.99** |
| Hand-written hard cases: no cues, statements, bank details, addresses | 130 | **98.5%** | 1 / 1 | 98.2% | 97.6% | **0.98** |
| Uncued names benchmark (frozen, never tuned on) | 56 | **94.6%** | 0 / 3 | 96.3% | 95.1% | **0.96** |
| **Held-out test reports (scored once, after the freeze)** | 604 | **100%** | 0 / 0 | 95.7% | 100% | **0.98** |

For context, a John Snow Labs benchmark found that two open-source AI-only tools (GLiNER PII and OpenPipe PII-Redact)
missed more than half of the personal details in clinical text ([article](https://medium.com/john-snow-labs/how-good-are-open-source-llm-based-de-identification-tools-in-a-medical-context-6600ddac6a0f)).
That is why PrivLab does not rely on an AI alone: rules, checksums and name lists on the device do most of the work,
and the AI is only a second check. Different data, so the numbers are not directly comparable.

**With the AI safety check** (Muse Spark Open reads the masked text once, after you confirm), the full system catches
**100% of the uncued names** (56 of 56; 3 reached the check unmasked and it caught them) and **98.5% of the hard
cases**. The two left are a vague contact ("line 3") and a road described by landmarks. The check also suggested 2
extra words in the hard-case negatives; you see its suggestions ticked on the De-identify step and can keep them.

The hard cases helped shape the rules, so they are a development score. The uncued benchmark was frozen before any
rule was tuned, so it is an honest one. The 50 held-out test reports were scored once, after the rules were frozen:
every one of the 604 personal details was hidden on the device and none reached the AI; 43 words were hidden that
were not personal (25 of them shown as "possible name" for you to check and untick), so precision there is 95.7%.

**What PrivLab keeps by default.** Clinical dates (collected, reported) and ages stay visible; only the date of birth is
hidden. They are often needed for analysis, but under the strictest US rule (HIPAA Safe Harbor) every date tied to a
person except the year, and any age over 89, counts as an identifier, and these scores do not count them. You can tick
"Other dates" in the privacy choices to hide all dates. Date shifting and "90+" ages are on the roadmap.

## Running the tests

```powershell
.venv\Scripts\python -m pytest -q                        # 260 tests, no AI calls (a fake model is used)
.venv\Scripts\python -m eval.score_deid --split hard     # de-identification score on the hard cases
```

## What's next

- Run the safety check fully in-house, so nothing leaves the organisation during de-identification.
- Date shifting (all of a patient's dates moved by the same number of days) and "90+" ages, for HIPAA Safe Harbor.
- Realistic stand-in names instead of tags, so a missed name hides among invented ones.
- A score on a public clinical benchmark (e.g. the n2c2 2014 de-identification set).
- Read scanned documents (OCR) on the device.
- More languages and name forms, including Chinese, Japanese and Korean characters.
