# OCR comparison: EasyOCR vs PaddleOCR

- Test images: 21 (arabic, carved, document, game, hard, numbers, screen)
- Accuracy = 100% − character error rate. Key facts = numbers/names the game depends on, read exactly.
- Time = average per image on CPU, after a warm-up run (model loading not included).

## Overall

| Engine | Preprocess | Accuracy | Key facts | Avg time / image |
|---|---|---|---|---|
| easyocr | enhance | 93.0% | 73% | 5.08s |
| easyocr | none | 93.4% | 77% | 3.26s |
| paddleocr | enhance | 98.6% | 98% | 5.90s |
| paddleocr | none | 99.5% | 100% | 4.34s |

## By category (accuracy)

| Engine | Preprocess | arabic | carved | document | game | hard | numbers | screen |
|---|---|---|---|---|---|---|---|---|
| easyocr | enhance | 93% | 91% | 99% | 95% | 87% | 100% | 99% |
| easyocr | none | 94% | 83% | 99% | 95% | 99% | 98% | 100% |
| paddleocr | enhance | 94% | 100% | 100% | 100% | 97% | 100% | 100% |
| paddleocr | none | 96% | 100% | 100% | 100% | 100% | 100% | 100% |

## Per image (accuracy · key facts)

| Image | easyocr / enhance | easyocr / none | paddleocr / enhance | paddleocr / none |
|---|---|---|---|---|
| 01_doc_clean — Clean typed memo (easy baseline) | 98% · 3/4 | 98% · 3/4 | 100% · 4/4 | 100% · 4/4 |
| 02_doc_small — Same memo, very small font | 98% · 3/3 | 98% · 3/3 | 100% · 3/3 | 100% · 3/3 |
| 03_report_serif — Serif font, official report | 98% · 2/3 | 98% · 2/3 | 100% · 3/3 | 100% · 3/3 |
| 04_receipt_mono — Receipt: monospace, numbers, prices | 100% · 4/4 | 99% · 4/4 | 100% · 4/4 | 100% · 4/4 |
| 05_numbers_ids — Phone, plate, ID numbers, codes | 100% · 4/4 | 96% · 3/4 | 100% · 4/4 | 100% · 4/4 |
| 06_dark_noisy_cctv — CCTV overlay: dark + heavy noise | 95% · 2/3 | 100% · 3/3 | 100% · 3/3 | 100% · 3/3 |
| 07_low_contrast — Faded grey text on grey paper | 61% · 0/2 | 100% · 2/2 | 98% · 2/2 | 98% · 2/2 |
| 08_blurred — Out-of-focus photo of a note | 82% · 1/2 | 95% · 1/2 | 100% · 2/2 | 100% · 2/2 |
| 09_rotated_photo — Label photographed at an angle (7 degrees) | 97% · 1/3 | 100% · 3/3 | 100% · 3/3 | 100% · 3/3 |
| 10_phone_chat — Phone screenshot: light text on dark | 99% · 2/3 | 100% · 3/3 | 100% · 3/3 | 100% · 3/3 |
| 11_italic_note — Slanted (italic) note on yellow paper | 100% · 3/3 | 100% · 3/3 | 100% · 3/3 | 100% · 3/3 |
| 12_tiny_plate — Tiny dark plate (130x38 px) - needs Enhance first | 100% · 1/1 | 100% · 1/1 | 86% · 1/1 | 100% · 1/1 |
| 13_arabic_doc — Arabic typed note (right to left) | 95% · 3/4 | 95% · 3/4 | 90% · 4/4 | 92% · 4/4 |
| 14_arabic_chat — Arabic chat screenshot | 90% · 3/3 | 93% · 3/3 | 98% · 2/3 | 100% · 3/3 |
| 15_game_transfer_notes — Real file from the USB test case | 95% · 4/4 | 95% · 3/4 | 100% · 4/4 | 100% · 4/4 |
| 22_carved_clear — Deep carving on a table, good light | 89% · 1/3 | 83% · 0/3 | 100% · 3/3 | 100% · 3/3 |
| 23_carved_shallow — Shallow/faint carving (barely visible) | 89% · 1/3 | 61% · 0/3 | 100% · 3/3 | 100% · 3/3 |
| 24_carved_knife_angled — Rough knife scratches, photo taken at an angle | 100% · 2/2 | 100% · 2/2 | 100% · 2/2 | 100% · 2/2 |
| 25_carved_dark — Carving photographed at night (dark + noise) | 79% · 1/2 | 63% · 1/2 | 100% · 2/2 | 100% · 2/2 |
| 26_carved_side_light_blur — Light from the side + out of focus | 88% · 1/2 | 88% · 1/2 | 100% · 2/2 | 100% · 2/2 |
| 27_carved_arabic — Arabic name + time carved into wood | 100% · 2/2 | 100% · 2/2 | 100% · 2/2 | 100% · 2/2 |

## Environment

- os: Windows-11-10.0.26200-SP0
- cpu: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel
- paddleocr: 3.7.0
- paddleocr_python: 3.12.10
- paddleocr_load_seconds: {'ar': 3.62, 'en': 1.4}
- easyocr: 1.7.2
- easyocr_python: 3.14.7
- easyocr_load_seconds: {'ar': 2.9, 'en': 2.08}
