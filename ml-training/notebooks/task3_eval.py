import os
os.chdir("/home/bradtab/git/FYP/ml-training/notebooks")

from ultralytics import YOLO
import sys
import io

# Capture stderr to check if warning is still present
import logging

# Task 2 baseline values
task2_map50   = 0.930
task2_map5095 = 0.760

model = YOLO("runs/detect/yolo26n_iter22/weights/best.pt")

# Capture output to check for warning
import contextlib
import io as _io

output_buffer = _io.StringIO()

results = model.val(
    data="../data/data.yaml",
    split="test",
    save_json=True,
    plots=True,
    project="runs/detect",
    name="yolo26n_iter22_eval_after_seg_fix"
)

map50   = results.box.map50
map5095 = results.box.map
per_cls = results.box.ap50
class_names = ["Sunglasses", "Keys", "Wallet"]

print("\n" + "=" * 60)
print("TASK 3 RESULTS (Step 3 — re-evaluation after seg fix):")
print("=" * 60)
print(f"  Files with segmentation annotations: 14")
print(f"  Total polygon lines converted: 19")
print(f"  mAP50 before fix: {task2_map50:.3f}  →  after fix: {map50:.3f}")
print(f"  mAP50-95 before fix: {task2_map5095:.3f}  →  after fix: {map5095:.3f}")
print(f"  Per-class AP50 (after fix):")
for i, name in enumerate(class_names):
    val = per_cls[i] if i < len(per_cls) else float('nan')
    print(f"    {name}: {val:.3f}")
print(f"  Warning still present: (check output above for 'len(segments)' message)")
