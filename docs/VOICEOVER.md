# Demo voiceover (draft, ~2 min, ~280 words)

Record with `python -m viewer.server --demo` and the built-in samples. Never show a real document.

| Time | Screen | Say |
|---|---|---|
| 0:00 | Start screen | Reports are full of useful data and full of personal details. So people retype the data by hand, or don't use AI at all. PrivLab fixes that. |
| 0:12 | Drop the Breast sample | I add a report. Before anything else, PrivLab finds the personal details on this device: names, ID numbers, phones, addresses. Nothing has been sent. |
| 0:25 | Review identifiers | I check the highlights and confirm. You're always in control: click to keep a word, select to hide one. |
| 0:38 | De-identify | Now every detail is a tag, like PERSON-1. Only this masked text goes to the AI, once, as a safety check for anything I missed. |
| 0:52 | Choose fields → Add a field | I want tumour size. I describe it in my own words; the AI drafts a precise definition, and I approve it. No report text is sent for this. |
| 1:08 | Extract | The AI reads the masked text and proposes each value. PrivLab checks every value against its quote in the report, so each one is traced to its source. I confirm. |
| 1:22 | Second report | And it learns. A name I hid on the first report is caught on this device the next time; a correction I make steers the next extraction. |
| 1:38 | Export | I export: a CSV with one row per report, ready for analysis, the masked text, or both. Files are built on this device. |
| 1:48 | Results card or README table | On our test sets, PrivLab catches 98% of hard identifiers before any API call, with no false alarms. And because the safety check runs on open weights, a hospital can host it itself. |
| 1:58 | Logo | PrivLab: reports in, structured data out, personal details stay home. |

Numbers to re-check before recording: replace the 1:48 figure with the held-out test result once it exists.
