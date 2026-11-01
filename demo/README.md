# Demo reports (synthetic)

Two invented Singapore breast-cancer reports for the demo video. No real
people, hospitals or addresses; NRICs are checksum-valid but fictional.

- report_1.txt: de-id learning (lower-case name) plus teach "tumour size".
  Expected tumour size: 2.1 cm.
- report_2.txt: the learned name is caught on the laptop; size is read.
  Expected tumour size: 1.7 cm.

De-id learning name (lower case, uncued, once per report): zola mensah.

Click path. Report 1: review -> second check finds the lower-case name ->
hide -> define "tumour size" -> correct the value to 2.1 by selecting it ->
Report 2: name caught on the laptop; tumour size read.
