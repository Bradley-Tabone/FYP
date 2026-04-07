import os
import glob

TEST_LBL = '/home/bradtab/git/FYP/ml-training/data/test/labels'
TEST_CACHE = '/home/bradtab/git/FYP/ml-training/data/test/labels.cache'

# ─── Step 1: Scan for segmentation annotation files ──────────────────────────
print("=" * 60)
print("TASK 3 — STEP 1: Scanning test/labels for segmentation lines")
print("=" * 60)

seg_files = {}  # filename -> list of (line_idx, line_content)

label_files = sorted(glob.glob(os.path.join(TEST_LBL, '*.txt')))
print(f"Total .txt files in test/labels: {len(label_files)}")

for fpath in label_files:
    seg_lines = []
    with open(fpath) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # segmentation: class + x1 y1 x2 y2 ... (>= 6 parts total, i.e., > 5 columns)
            if len(parts) > 5:
                seg_lines.append((i, line))
    if seg_lines:
        seg_files[os.path.basename(fpath)] = seg_lines

print(f"\nFiles with segmentation annotations: {len(seg_files)}")
total_seg_lines = sum(len(v) for v in seg_files.values())
print(f"Total polygon lines: {total_seg_lines}")
print()
for fname, lines in seg_files.items():
    print(f"  {fname}: {len(lines)} seg line(s)")

# ─── Step 2: Convert polygon to bounding box ─────────────────────────────────
print("\n" + "=" * 60)
print("TASK 3 — STEP 2: Converting polygon lines to bounding boxes")
print("=" * 60)

def polygon_to_bbox(parts):
    """
    parts: ['class', 'x1', 'y1', 'x2', 'y2', ...]
    Returns: 'class cx cy w h'
    """
    cls = parts[0]
    coords = [float(v) for v in parts[1:]]
    xs = coords[0::2]  # even indices: x coords
    ys = coords[1::2]  # odd indices: y coords

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w  = xmax - xmin
    h  = ymax - ymin

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))

    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

converted_total = 0

for fname in seg_files:
    fpath = os.path.join(TEST_LBL, fname)
    with open(fpath) as f:
        raw_lines = f.readlines()

    new_lines = []
    n_converted = 0
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        parts = stripped.split()
        if len(parts) > 5:
            new_line = polygon_to_bbox(parts)
            new_lines.append(new_line + '\n')
            n_converted += 1
        else:
            new_lines.append(line if line.endswith('\n') else line + '\n')

    with open(fpath, 'w') as f:
        f.writelines(new_lines)

    print(f"  Converted {n_converted} line(s) in {fname}")
    converted_total += n_converted

print(f"\nTotal polygon lines converted: {converted_total}")

# Delete cache so YOLO re-scans
if os.path.exists(TEST_CACHE):
    os.remove(TEST_CACHE)
    print(f"Removed cache: {TEST_CACHE}")

print("\n" + "=" * 60)
print("TASK 3 — STEP 2 COMPLETE")
print("=" * 60)
print(f"Files fixed: {len(seg_files)}")
print(f"Lines converted: {converted_total}")
