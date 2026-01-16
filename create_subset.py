import os
import shutil
import random
import yaml
from collections import defaultdict

# Config
SOURCE_DIR = "data/pcb-defect-dataset"
DEST_DIR = "data/pcb-small"
NUM_IMAGES = 500  # Number of images to use

def get_balanced_subset(split, count):
    src_img_dir = os.path.join(SOURCE_DIR, split, "images")
    src_lbl_dir = os.path.join(SOURCE_DIR, split, "labels")
    
    # Map classes to files
    class_to_files = defaultdict(list)
    all_files = [f for f in os.listdir(src_lbl_dir) if f.endswith('.txt')]
    
    print(f"Scanning {len(all_files)} files in {split} for class balancing...")
    
    for lbl_file in all_files:
        try:
            with open(os.path.join(src_lbl_dir, lbl_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        cls = int(line.split()[0])
                        class_to_files[cls].append(lbl_file)
        except:
            pass

    # Round-robin selection to ensure balance
    selected_files = set()
    classes = list(class_to_files.keys())
    
    while len(selected_files) < count and any(class_to_files.values()):
        for cls in classes:
            if class_to_files[cls]:
                # Pick a random file from this class
                f = random.choice(class_to_files[cls])
                if f not in selected_files:
                    selected_files.add(f)
                # Remove from pool to avoid infinite loops if we run out of unique files for a class
                # (Simple approach: just keep picking until we hit count or run out)
                # Better: just pick random, check unique.
                
                if len(selected_files) >= count:
                    break
        
        # If we cycled through all classes and didn't add anything (all duplicates), 
        # just pick random remaining files to fill quota
        if len(selected_files) < count:
             remaining = [f for f in all_files if f not in selected_files]
             if not remaining: break
             selected_files.add(random.choice(remaining))

    print(f"Selected {len(selected_files)} balanced files for {split}.")
    
    # Copy files
    dst_img_dir = os.path.join(DEST_DIR, split, "images")
    dst_lbl_dir = os.path.join(DEST_DIR, split, "labels")
    
    for lbl_file in selected_files:
        # Copy Label
        shutil.copy(os.path.join(src_lbl_dir, lbl_file), os.path.join(dst_lbl_dir, lbl_file))
        
        # Copy Image (try jpg, png, jpeg)
        base_name = os.path.splitext(lbl_file)[0]
        for ext in ['.jpg', '.png', '.jpeg']:
            img_name = base_name + ext
            src_img = os.path.join(src_img_dir, img_name)
            if os.path.exists(src_img):
                shutil.copy(src_img, os.path.join(dst_img_dir, img_name))
                break

# Create directories
if os.path.exists(DEST_DIR):
    shutil.rmtree(DEST_DIR)

os.makedirs(os.path.join(DEST_DIR, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "val", "labels"), exist_ok=True)

# Copy Data
get_balanced_subset("train", NUM_IMAGES)
get_balanced_subset("val", 50) # Balanced validation set

# Create data_small.yaml
data_yaml = {
    'train': os.path.abspath(os.path.join(DEST_DIR, "train", "images")),
    'val': os.path.abspath(os.path.join(DEST_DIR, "val", "images")),
    'nc': 7,
    'names': ['Open', 'Short', 'Mouse_bite', 'Spur', 'Missing_hole', 'Spurious_copper', 'Burnt']
}

with open('data_small.yaml', 'w') as f:
    yaml.dump(data_yaml, f)

print("Balanced subset created and data_small.yaml generated.")
