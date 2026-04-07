import os
import sys

os.chdir("/home/bradtab/git/FYP/ml-training/notebooks")

from ultralytics import YOLO

model = YOLO("runs/detect/yolo26n_iter22/weights/best.pt")
results = model.val(
    data="../data/data.yaml",
    split="test",
    save_json=True,
    plots=True,
    project="runs/detect",
    name="yolo26n_iter22_eval"
)

print("\n" + "="*60)
print("TASK 1 RESULTS:")
print("="*60)

map50 = results.box.map50
map5095 = results.box.map

# Per-class AP50
class_names = ["Sunglasses", "Keys", "Wallet"]
per_class_ap50 = results.box.ap50  # per-class AP50

print(f"  mAP50:    {map50:.3f}")
print(f"  mAP50-95: {map5095:.3f}")
print(f"  Per-class AP50:")
for i, name in enumerate(class_names):
    if i < len(per_class_ap50):
        print(f"    {name}: {per_class_ap50[i]:.3f}")
    else:
        print(f"    {name}: N/A")

print(f"\n  Sunglasses={per_class_ap50[0]:.3f}, Keys={per_class_ap50[1]:.3f}, Wallet={per_class_ap50[2]:.3f}" if len(per_class_ap50) >= 3 else "  Per-class AP50 incomplete")
