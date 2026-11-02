# Demo reports (synthetic)

Invented reports for the demo and the video. No real people, hospitals or addresses; NRICs are checksum-valid but
fictional; postcodes and phone numbers are made up.

| File | Sample chip | Shows |
|---|---|---|
| `report_1.txt` | **Breast** | De-id learning: a lower-case name (`zola mensah`) the rules miss; teach "tumour size" (2.1 cm); pT2 pN0(sn) |
| `lung.txt` / `lung.pdf` | **Lung** | Synoptic layout: "Surname, Given names", DOB / Age, a letter-spaced barcode, an address over two lines, a capitals letterhead, middle initials; tumour size 3.4 cm, 2 of 14 nodes, LVI present, pT2a pN1, biomarkers (EGFR, ALK, PD-L1) |
| `colon.txt` | **Colon** | Narrative layout: a relative with a mobile number; tumour size 4.2 cm, 3 of 22 nodes, margins clear (6 mm), LVI present, pT3 pN1b, mismatch repair results (a list field) |
| `report_2.txt` | (drop the file) | Learning, part 2: the name taught on the Breast report is now caught on the device; size 1.7 cm; pT1c pN0(sn) |

Click path for the video: **Breast** sample → review → the safety check finds `zola mensah` → hide it → add the
field "tumour size" → correct it to 2.1 by selecting it → finish. Drop `report_2.txt`: the name is caught on the
device. Drop `lung.pdf`: the PDF badge, **Original ↗**, and every header identifier masked. **All 3** adds the
three samples as a batch for **Extract all**.
