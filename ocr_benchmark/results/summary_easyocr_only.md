# OCR comparison: EasyOCR vs PaddleOCR

- Test images: 15 (arabic, document, game, hard, numbers, screen)
- Accuracy = 100% − character error rate. Key facts = numbers/names the game depends on, read exactly.
- Time = average per image on CPU, after a warm-up run (model loading not included).

## Overall

| Engine | Preprocess | Accuracy | Key facts | Avg time / image |
|---|---|---|---|---|
| easyocr | enhance | 93.9% | 78% | 4.45s |
| easyocr | none | 97.8% | 87% | 2.82s |

## By category (accuracy)

| Engine | Preprocess | arabic | document | game | hard | numbers | screen |
|---|---|---|---|---|---|---|---|
| easyocr | enhance | 93% | 99% | 95% | 87% | 100% | 99% |
| easyocr | none | 94% | 99% | 95% | 99% | 98% | 100% |

## Per image (accuracy · key facts)

| Image | easyocr / enhance | easyocr / none |
|---|---|---|
| 01_doc_clean — Clean typed memo (easy baseline) | 98% · 3/4 | 98% · 3/4 |
| 02_doc_small — Same memo, very small font | 98% · 3/3 | 98% · 3/3 |
| 03_report_serif — Serif font, official report | 98% · 2/3 | 98% · 2/3 |
| 04_receipt_mono — Receipt: monospace, numbers, prices | 100% · 4/4 | 99% · 4/4 |
| 05_numbers_ids — Phone, plate, ID numbers, codes | 100% · 4/4 | 96% · 3/4 |
| 06_dark_noisy_cctv — CCTV overlay: dark + heavy noise | 95% · 2/3 | 100% · 3/3 |
| 07_low_contrast — Faded grey text on grey paper | 61% · 0/2 | 100% · 2/2 |
| 08_blurred — Out-of-focus photo of a note | 82% · 1/2 | 95% · 1/2 |
| 09_rotated_photo — Label photographed at an angle (7 degrees) | 97% · 1/3 | 100% · 3/3 |
| 10_phone_chat — Phone screenshot: light text on dark | 99% · 2/3 | 100% · 3/3 |
| 11_italic_note — Slanted (italic) note on yellow paper | 100% · 3/3 | 100% · 3/3 |
| 12_tiny_plate — Tiny dark plate (130x38 px) - needs Enhance first | 100% · 1/1 | 100% · 1/1 |
| 13_arabic_doc — Arabic typed note (right to left) | 95% · 3/4 | 95% · 3/4 |
| 14_arabic_chat — Arabic chat screenshot | 90% · 3/3 | 93% · 3/3 |
| 15_game_transfer_notes — Real file from the USB test case | 95% · 4/4 | 95% · 3/4 |

## Environment

- python: 3.14.7
- os: Windows-11-10.0.26200-SP0
- cpu: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel
- easyocr: 1.7.2
- easyocr_load_seconds: {'ar': 63.58, 'en': 4.85}
