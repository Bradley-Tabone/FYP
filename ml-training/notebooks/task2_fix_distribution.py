import os
import shutil
import random
import json

random.seed(42)

TRAIN_IMG = '/home/bradtab/git/FYP/ml-training/data/train/images'
TRAIN_LBL = '/home/bradtab/git/FYP/ml-training/data/train/labels'
VALID_IMG = '/home/bradtab/git/FYP/ml-training/data/valid/images'
VALID_LBL = '/home/bradtab/git/FYP/ml-training/data/valid/labels'
TEST_IMG  = '/home/bradtab/git/FYP/ml-training/data/test/images'
TEST_LBL  = '/home/bradtab/git/FYP/ml-training/data/test/labels'

CLASS_NAMES = {0: 'Sunglasses', 1: 'Keys', 2: 'Wallet'}

# ─── Step 1: Identify supplementary images ───────────────────────────────────
print("=" * 60)
print("STEP 1: Identifying supplementary images in train")
print("=" * 60)

supplementary = []  # (image_filename, dominant_class)

train_images = [
    f for f in os.listdir(TRAIN_IMG)
    if f.startswith('2026') and '.rf.' in f and '_aug' not in f and ':Zone' not in f
]

for img in sorted(train_images):
    stem = os.path.splitext(img)[0]
    label_path = os.path.join(TRAIN_LBL, stem + '.txt')

    if not os.path.exists(label_path):
        continue

    with open(label_path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if not lines:
        # Background image — assign to a class based on image context
        # Skip empty labels as they don't contribute to stratified sampling
        continue

    class_freq = {}
    for line in lines:
        cls = int(line.split()[0])
        class_freq[cls] = class_freq.get(cls, 0) + 1

    dominant = max(class_freq, key=class_freq.get)
    supplementary.append((img, dominant))

print(f"Supplementary images found in train: {len(supplementary)}")

# Count by class
class_pools = {0: [], 1: [], 2: []}
for img, cls in supplementary:
    class_pools[cls].append(img)

print("Class distribution:")
for cls, imgs in class_pools.items():
    print(f"  {CLASS_NAMES[cls]}: {len(imgs)}")

# ─── Step 2: Select images to move ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Selecting images for valid and test (stratified)")
print("=" * 60)

# Strategy: use all sunglasses (49), split evenly between valid/test
# Keys: 33 to valid, 33 to test
# Wallets: 34 to valid, 34 to test
# Sunglasses: 25 to valid, 24 to test (all 49)

n_sun = len(class_pools[0])   # 49
n_keys = len(class_pools[1])  # 169
n_wallets = len(class_pools[2])  # 151

# Shuffle each class pool
for cls in class_pools:
    random.shuffle(class_pools[cls])

# How many per class to move to valid and test
# Maximise sunglasses usage: split 49 into 25/24
sun_to_valid = min(25, n_sun // 2 + n_sun % 2)
sun_to_test  = min(24, n_sun - sun_to_valid)

# Keys and wallets: aim for ~33-34 each to reach ~100 per split
keys_to_valid   = 33
keys_to_test    = 33
wallet_to_valid = 34
wallet_to_test  = 34

to_valid = (
    class_pools[0][:sun_to_valid] +
    class_pools[1][:keys_to_valid] +
    class_pools[2][:wallet_to_valid]
)
to_test = (
    class_pools[0][sun_to_valid:sun_to_valid + sun_to_test] +
    class_pools[1][keys_to_valid:keys_to_valid + keys_to_test] +
    class_pools[2][wallet_to_valid:wallet_to_valid + wallet_to_test]
)

print(f"Images selected for valid: {len(to_valid)}")
print(f"  Sunglasses: {sun_to_valid}, Keys: {keys_to_valid}, Wallets: {wallet_to_valid}")
print(f"Images selected for test: {len(to_test)}")
print(f"  Sunglasses: {sun_to_test}, Keys: {keys_to_test}, Wallets: {wallet_to_test}")

# ─── Step 3: Move image and label files ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Moving files")
print("=" * 60)

def move_image_and_label(img_name, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    stem = os.path.splitext(img_name)[0]
    src_img = os.path.join(src_img_dir, img_name)
    dst_img = os.path.join(dst_img_dir, img_name)
    src_lbl = os.path.join(src_lbl_dir, stem + '.txt')
    dst_lbl = os.path.join(dst_lbl_dir, stem + '.txt')

    moved = {'img': False, 'lbl': False}

    if os.path.exists(src_img):
        shutil.move(src_img, dst_img)
        moved['img'] = True
    else:
        print(f"  WARNING: image not found: {src_img}")

    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)
        moved['lbl'] = True
    else:
        print(f"  WARNING: label not found: {src_lbl}")

    return moved

moved_to_valid = 0
moved_to_test  = 0

for img in to_valid:
    result = move_image_and_label(img, TRAIN_IMG, TRAIN_LBL, VALID_IMG, VALID_LBL)
    if result['img']:
        moved_to_valid += 1

for img in to_test:
    result = move_image_and_label(img, TRAIN_IMG, TRAIN_LBL, TEST_IMG, TEST_LBL)
    if result['img']:
        moved_to_test += 1

print(f"Moved to valid: {moved_to_valid}")
print(f"Moved to test:  {moved_to_test}")

# ─── Step 4: Verify data.yaml ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Verifying data.yaml paths")
print("=" * 60)

data_yaml_path = '/home/bradtab/git/FYP/ml-training/data/data.yaml'
with open(data_yaml_path) as f:
    content = f.read()
print("data.yaml content:")
print(content)
print("Paths check: train/images, valid/images, test/images should exist")
for split in ['train', 'valid', 'test']:
    img_dir = f'/home/bradtab/git/FYP/ml-training/data/{split}/images'
    lbl_dir = f'/home/bradtab/git/FYP/ml-training/data/{split}/labels'
    n_img = len([f for f in os.listdir(img_dir) if not f.endswith(':Zone.Identifier')])
    n_lbl = len([f for f in os.listdir(lbl_dir) if not f.endswith(':Zone.Identifier')])
    print(f"  {split}: {n_img} images, {n_lbl} labels")

# Delete old labels caches so YOLO re-scans
for split in ['train', 'valid', 'test']:
    cache_path = f'/home/bradtab/git/FYP/ml-training/data/{split}/labels.cache'
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Removed cache: {cache_path}")

print("\n" + "=" * 60)
print("TASK 2 STEP 1-4 COMPLETE")
print("=" * 60)
print(f"Supplementary images found in train: {len(supplementary)}")
print(f"Moved to valid: {moved_to_valid} (Sunglasses={sun_to_valid}, Keys={keys_to_valid}, Wallet={wallet_to_valid})")
print(f"Moved to test:  {moved_to_test} (Sunglasses={sun_to_test}, Keys={keys_to_test}, Wallet={wallet_to_test})")
