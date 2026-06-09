# Review of BT_FYP_Report_2026.pdf

Reviewed PDF: `C:\Users\Bradley\OneDrive\Desktop\FYP submission\BT_FYP_Report_2026.pdf`

Review date: 2026-06-03

## Overall Assessment

The report is in good shape structurally. The thesis has a clear problem statement, a defensible comparison between YOLO26, RT-DETR, and RF-DETR, and a sensible conclusion: YOLO26n is the only evaluated model that meets the real-time on-device target. The argument generally flows from user need, to model/dataset design, to Android deployment, to evaluation.

The main risks before submission are not the broad story, but several correctness and consistency issues in the results and appendices. The most important one is a metric-label error around Table 4.5: the per-class values listed as AP@50 are actually per-class mAP@50-95 values from Ultralytics `metrics.box.maps`. This should be fixed before submission because it affects the interpretation of the results chapter, future work, and conclusion.

## Must Fix Before Submission

### 1. Table 4.5 Metric Label Is Incorrect

Location: PDF pages 42-43, thesis pages 30-31.

The thesis says Table 4.5 reports per-class AP@50:

- Keys: 0.7643
- Sunglasses: 0.7135
- Wallet: 0.7666
- Glasses: 0.8219

However, these four values average to about 0.7666, exactly matching the reported overall mAP@50-95. The notebook output confirms the source of the issue: the code prints `AP50=` while using `metrics.box.maps`, which in Ultralytics represents per-class mAP@50-95, not per-class AP@50.

Fix options:

- Easiest fix: change Table 4.5 title and column from `AP@50` to `AP@50-95` or `mAP@50-95`, and update all surrounding prose.
- Better fix: recompute and report true per-class AP@50 values if you want the section to discuss AP@50 specifically.

Affected prose to update:

- "Table 4.5 reports per-class AP@50..."
- "The Glasses class achieved AP@50 of 0.8219..."
- "Its AP@50 of 0.7643..."
- "Wallet achieved AP@50 of 0.7666..."
- Future Work: "Its test-set AP@50 score of 0.7643..."

Suggested wording if keeping the current numbers:

> Table 4.5 reports per-class mAP@50-95 for the YOLO26n iter5 INT8 TFLite model on the test split. The overall INT8 TFLite model achieved mAP@50 of 0.9536 and mAP@50-95 of 0.7666.

### 2. Revalidate the Test Split Used for Final YOLO26n Metrics

Location: PDF page 38, thesis page 26; notebook evidence in `Iteration2.ipynb`.

The report says the Iteration 2 model was evaluated on a test split of 408 images. The current folder `ml-training/data/preprocessed/test` does contain 408 image files and 408 label files. However, the YOLO26n INT8 TFLite validation output copied in the notebook shows a cached scan of 376 images and 428 boxes:

> `Scanning ... 376 images, 22 backgrounds...`

This does not automatically invalidate the result, but it is a reproducibility issue. Before submission, rerun the final YOLO26n INT8 validation after deleting stale `.cache` files, then make sure the PDF, notebook, and current dataset all agree on:

- number of test images
- number of test object instances
- mAP@50
- mAP@50-95
- per-class metrics

### 3. Appendix A Contradicts the Results About the Runtime Delegate

Location: PDF pages 52-53, thesis pages 40-41.

Appendix A says:

> NMS-free so the full graph runs on the mobile GPU delegate

and later:

> Inference uses the LiteRT GPU delegate where available and falls back to CPU otherwise.

But the Results chapter says the GPU delegate rejected the INT8 graph and YOLO26n ran through NNAPI on the Hexagon DSP. The code also tries GPU, then NNAPI, then CPU.

Fix:

> Exports YOLO26n to a calibrated INT8 TFLite model. The Android app attempts GPU execution first, then NNAPI, then CPU. On the Samsung Galaxy S23 Ultra, the deployed INT8 model was executed through NNAPI on the Hexagon DSP.

### 4. Offline / No-Internet Claim Needs Qualification

Location: Abstract, Introduction, Methodology page 33.

The model inference itself is offline, but the speech-recognition claim is too strong as written. The app uses Android `SpeechRecognizer.createSpeechRecognizer(...)` with `ACTION_RECOGNIZE_SPEECH`, but does not set or verify an on-device recognizer. Android's official docs say the recognizer implementation may use remote servers, and even `EXTRA_PREFER_OFFLINE` may have no effect depending on implementation.

Fix the claim unless you have tested and documented offline speech recognition on the target device.

Safer wording:

> The object-detection and guidance pipeline runs without internet connectivity. Speech recognition uses Android's SpeechRecognizer API and depends on the recognition service available on the device.

If you want the full "no internet" claim, implement/check on-device recognition explicitly and document the tested device setting.

### 5. Front-Matter TOC Page Numbers Are Wrong for Multi-Page Entries

Location: Contents, PDF pages 4-6.

The visible roman numeral page numbering is correct, but some Contents entries point to the final page of a multi-page front-matter section instead of the start:

- `Contents` is shown as page `v`, but it starts on `iii`.
- `List of Abbreviations` is shown as `x`, but it starts on `viii`.

Likely LaTeX cause: `\addcontentsline` or hyperlink anchors are placed after the generated section/list instead of before it.

Fix by placing the anchor and contents entry before the multi-page front-matter command, for example using `\phantomsection` before `\addcontentsline`.

## High Priority Flow and Structure Fixes

### Results Chapter

Location: PDF pages 38-44.

The final model selection is logically sound, but the metric narrative needs tightening:

- Table 4.3 mixes mAP@50 and RF-DETR EMA mAP@50-95, then Table 4.6 lists RF-DETR Nano `Val mAP@50 = 0.962`. This looks inconsistent unless that 0.962 comes from a separate comparable evaluation. Add a footnote or use only comparable metrics in Table 4.6.
- Table 4.1 caption says "Roboflow base dataset with custom images", but Iteration 1 is described as the Roboflow base dataset before the custom image expansion. Remove "with custom images" unless custom images were actually included.
- "RF-DETR Base achieved the lowest mAP@50 ... indicating that the model was trained for only 50 epochs..." is too causal. A result does not prove the cause. Use "suggesting that 50 epochs may have been insufficient..." or "one likely factor is..."
- "all candidate models ... using identical training configurations" is too strong across YOLO, RT-DETR, and RF-DETR. Use "same dataset and epoch budget where possible" or "controlled comparison conditions".

### Appendix B Layout

Location: PDF pages 55-58, thesis pages 43-46.

Appendix B currently introduces B.2, immediately skips to B.3, then places the B.2 table on the next page. This makes it look like the Model Training subsection is missing.

Fix:

- Put the B.2 paragraph/table immediately after the B.2 heading.
- Start B.3 only after Table B.2.
- Consider forcing a page break before B.2 if needed.

Page 56 also has a large blank area before Table B.2, likely due to float placement.

### Appendix A Reproducibility Details

Location: PDF pages 50-54.

Fix these:

- Use `FindME` consistently, not `FindMe`.
- Capitalize `Google Drive`.
- Replace "FYP source-code submission folder" with a real visible URL or clear submission-media reference.
- The copy command on page 52 is broken across lines and would not run as pasted. Put it in a code block as one command or use a proper PowerShell line continuation.
- It says Iteration1 compares seven detector variants, but the methodology/results list eight.

## Page-Level Corrections

### Front Matter

- Page 2: "three commonly used items being sunglasses, keys, and a wallet" -> "three commonly used items: sunglasses, keys, and a wallet"
- Page 3: "Special thanks also goes out" -> "Special thanks also go to"
- Page 3: "turned it into a fun one" is informal for a thesis. Consider "made the process more enjoyable."
- Page 7: "during aYOLO26n" -> "during a YOLO26n"
- Page 8: "bestvalidation" -> "best validation"
- Page 8: "Table 4.5 YOLO26n..." -> "Table 4.5  YOLO26n..." or add missing spacing in source.

### Introduction

- Page 13: "Bluetooth trackers, such as Apple AirTag and Tile that require..." -> add a comma after "Tile" or restructure.
- Page 13: "motion sensors namely" -> "motion sensors, namely"
- Page 13: "leftthe camera frame" -> "left the camera frame"
- Page 14: "lefton a wooden table" -> "left on a wooden table"
- Page 14: "Each present" -> "Each presents"
- Page 14: "potentially frustrating to and time-consuming for" -> "potentially frustrating and time-consuming for"
- Page 15: switch proposal-style future tense to completed-project tense:
  - "will be evaluated" -> "were evaluated"
  - "will be deployed" -> "was deployed"
  - "will be based" -> "was based"

### Background and Literature Review

- Page 16: The first paragraph repeats "what objects are in an image and where..." twice. Tighten it.
- Page 17: "YOLO26n is the nano variant deployed in this project ." -> remove space before period.
- Page 19: "trade-offbetween" -> "trade-off between"
- Page 19: "A mid-range Android device..." conflicts with the Samsung Galaxy S23 Ultra, which is not mid-range. Use "A mobile Android device..." unless you are discussing a specific mid-range deployment target.
- Page 21: "up till YOLO11" -> "up to YOLO11"
- Page 21: "recognized" mixes US spelling with UK-style "quantisation/colour/organisation"; use "recognised" if keeping UK style.
- Page 22: "hence, confirming" -> "thereby confirming" or "which confirms"
- Page 22: "utilizing" -> "using" or "utilising"
- Page 23: "most times seen as flat objects" -> "often viewed as flat, low-profile objects"
- Page 25: "Transformer architecture models" -> "Transformer-based models"
- Page 25: "trade-offon" -> "trade-off on"

### Methodology

- Page 26: Opening paragraph repeats the two-iteration description. Combine into one cleaner paragraph.
- Page 26: "training of the models was performed" -> "model training was performed"
- Page 26: "As a results" -> "As a result"
- Page 26: The sentence introducing the Glasses class is duplicated.
- Page 27: "taken in real life" -> "captured in real domestic environments"
- Page 27: "each object class was taken" -> "images of each class were captured"
- Page 27: "clear lenses versus those tinted lenses" -> "clear lenses versus tinted lenses"
- Page 27: "utilized" / "recognized" -> align spelling style.
- Page 28: "due to both the fact..." -> "because it created near-duplicates and did not add new variation between epochs"
- Page 30: `model.export(format=`tflite', ...)` has mismatched quotes. Use `model.export(format='tflite', ...)`.
- Page 33: "turning offduring" -> "turning off during"
- Page 33: "SpeechRecognizer API, which processes audio locally on the device" -> qualify or verify; see offline speech note above.
- Page 34: "turn offuntil" -> "turn off until"
- Page 37: "in-real time guidance" -> "real-time guidance"
- Page 37: "eyes closed to simulate the experience of the target user" -> consider "to approximate a non-visual search condition"; it is not equivalent to visually impaired user testing.

### Results and Evaluation

- Page 38: "seperately" -> "separately"
- Page 38: "target frame rate and latency ... is" -> "are"
- Page 39: Table 4.1 caption likely should remove "with custom images".
- Page 39: "reduces complexity of the exported model from inference" -> "reduces the complexity of the exported inference graph"
- Page 39: Avoid causal overclaim about RF-DETR Base and 50 epochs.
- Page 40: "ExtendedTraining" -> "Extended Training"
- Page 40: Make sure Table 4.3 and Table 4.6 use comparable metrics or clearly label the exception.
- Page 41: "however the deployment behaviour" -> "however, the deployment behaviour"
- Page 41: "ONNX ONNX" appears because table text repeats the format/runtime term; visually it is acceptable but could be cleaner as "ONNX Runtime / XNNPACK".
- Pages 42-43: Fix Table 4.5 metric labels and all related prose.
- Page 43: "Their small physical size" refers to Keys, but "Keys" is a class; use "Keys are physically small..." or "Keys often occupy..."
- Page 43: "reflections obscured the vision of the target keys" -> "reflections obscured the keys"
- Page 44: Table 4.6 RF-DETR Nano metric needs clarification because Table 4.3 says RF-DETR Nano was not directly comparable.

### Future Work and Conclusion

- Page 45: "torch lights from the device" -> "torch light from the device"
- Page 45: "recognize" -> "recognise" if keeping UK style.
- Page 45: "iOS devices..." is useful, but the project is Android. Lead with ARCore Depth API, then mention LiDAR as cross-platform context only if needed.
- Page 46: "visually-impaired" is used with a hyphen inconsistently. Prefer "visually impaired" as an adjective phrase unless your institution style guide says otherwise.
- Page 46: Avoid "Hence" as a sentence opener here; use "Therefore" or merge the sentence.
- Page 47: "models trained, evaluated, and deployed ... were YOLO26, RT-DETR-l, and RF-DETR Nano" -> only YOLO26n was deployed. Use "The models trained and evaluated were..., and YOLO26n was deployed."
- Page 47: "unable to be performed" -> "could not be performed"

### References

All citations [1]-[18] are present in the reference list, and every reference is cited.

Potential source improvement:

- Your YOLO26 claims are broadly consistent with current Ultralytics docs and the Sapkota/Karkee overview. However, Ultralytics now cites an official YOLO26 technical paper, "Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models" (arXiv:2606.03748). If submission timing permits, consider citing the official paper alongside or instead of the overview paper.
- RF-DETR source [10] is correctly aligned with the ICLR 2026/OpenReview/arXiv paper.
- Source [11] exists and supports the edge-device/runtime-comparison framing.

Sources checked:

- Ultralytics YOLO26 docs: https://docs.ultralytics.com/models/yolo26/
- Sapkota and Karkee YOLO26 overview: https://arxiv.org/abs/2510.09653
- RF-DETR arXiv: https://arxiv.org/abs/2511.09554
- RF-DETR ICLR 2026 OpenReview PDF: https://openreview.net/pdf/6c979474ef7804b04ad00f0b04b7f3311e0a6719.pdf
- Suchy and Turcanik Scientific Reports article: https://www.nature.com/articles/s41598-026-46453-6
- Android RecognizerIntent docs: https://developer.android.com/reference/kotlin/android/speech/RecognizerIntent

## Style Consistency

The report mostly uses UK style:

- fulfilment
- organised
- quantisation
- colour
- behaviour
- localisation
- labelled

But there are US-style forms mixed in:

- recognized
- recognize
- utilizing
- localization
- centered

Recommendation: use UK style throughout:

- recognised
- recognise
- using/utilising
- localisation
- centred

## Flow Assessment

Strong areas:

- The introduction motivates the user problem well.
- The literature review does not feel detached; it sets up why YOLO, RT-DETR, and RF-DETR were compared.
- The methodology gives enough detail for a technical reader to understand the pipeline.
- The results chapter makes the practical deployment argument clearly: runtime support matters as much as model accuracy.

Areas to tighten:

- The report sometimes says what will be done, especially in Chapter 1 and early Chapter 3, even though the work is complete.
- The dataset narrative has repeated statements about the Glasses class.
- The results chapter needs stricter separation between validation metrics, test metrics, mAP@50, and mAP@50-95.
- The appendices should be brought into line with the final deployment facts from Chapter 4.

## Suggested Final Pass Checklist

1. Fix Table 4.5 metric label/prose.
2. Rerun final YOLO26n INT8 validation after clearing stale dataset cache files.
3. Reconcile 376-image vs 408-image test split references.
4. Correct Appendix A GPU/NNAPI wording.
5. Qualify the no-internet/offline speech-recognition claim.
6. Fix TOC entries for Contents and List of Abbreviations.
7. Correct spacing typos and spelling consistency.
8. Reflow Appendix B so B.2 content appears before B.3.
9. Regenerate the PDF and check Contents/List of Figures/List of Tables again.
10. Do a final visual pass over pages with tables and appendices.

