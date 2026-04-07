import os
os.chdir("/home/bradtab/git/FYP/ml-training/notebooks")

from ultralytics import YOLO

# Task 1 baseline values
task1_map50    = 0.901
task1_map5095  = 0.710

model = YOLO("runs/detect/yolo26n_iter22/weights/best.pt")
results = model.val(
    data="../data/data.yaml",
    split="test",
    save_json=True,
    plots=True,
    project="runs/detect",
    name="yolo26n_iter22_eval_after_redist"
)

map50    = results.box.map50
map5095  = results.box.map
per_cls  = results.box.ap50
class_names = ["Sunglasses", "Keys", "Wallet"]

print("\n" + "=" * 60)
print("TASK 2 RESULTS (Step 5 — re-evaluation after redistribution):")
print("=" * 60)
print(f"  mAP50 before: {task1_map50:.3f}  →  after: {map50:.3f}")
print(f"  mAP50-95 before: {task1_map5095:.3f}  →  after: {map5095:.3f}")
print(f"  Per-class AP50 (after):")
for i, name in enumerate(class_names):
    val = per_cls[i] if i < len(per_cls) else float('nan')
    print(f"    {name}: {val:.3f}")
