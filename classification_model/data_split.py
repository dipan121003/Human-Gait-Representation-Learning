import os
import shutil
import random
from pathlib import Path

# === CONFIGURATION ===
SOURCE_DIR = Path("/home/teaching/Documents/G28_dl/cs671-p07/IMU_Dataset/Data_semi_processed")
TARGET_ROOT = Path("/home/teaching/Documents/G28_dl/cs671-p07/IMU_Dataset/Dataset/Self_supervise_data2")
SPLITS = ["train", "val", "test"]
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42

# === STEP 1: Gather all CSV files ===
print("🔍 Collecting all .csv files from dataset hierarchy...")
all_files = []
skipped = []

for root, _, files in os.walk(SOURCE_DIR):
    for f in files:
        file_path = Path(root) / f
        if file_path.suffix == ".csv" and file_path.is_file():
            all_files.append(file_path)
        elif file_path.suffix == ".csv":
            skipped.append(file_path)

print(f"✅ Found {len(all_files)} CSV files.")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} malformed or directory-like .csv entries.")

# === STEP 2: Shuffle and Split ===
random.seed(RANDOM_SEED)
random.shuffle(all_files)

total = len(all_files)
num_train = int(total * SPLIT_RATIOS["train"])
num_val = int(total * SPLIT_RATIOS["val"])
num_test = total - num_train - num_val

splits = {
    "train": all_files[:num_train],
    "val": all_files[num_train:num_train + num_val],
    "test": all_files[num_train + num_val:]
}

# === STEP 3: Copy files flat ===
print("\n📦 Copying files directly into train/val/test folders (no hierarchy)...")
for split in SPLITS:
    split_dir = TARGET_ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, file_path in enumerate(splits[split]):
        new_filename = f"{split}_{idx+1}.csv"
        dest_file = split_dir / new_filename
        shutil.copy2(file_path, dest_file)

# === Final Summary ===
print("\n✅ Completed file flattening and splitting:")
for split in SPLITS:
    count = len(splits[split])
    print(f"📂 {split.capitalize()}: {count} files")

if skipped:
    print(f"\n⚠️ {len(skipped)} malformed entries skipped.")
