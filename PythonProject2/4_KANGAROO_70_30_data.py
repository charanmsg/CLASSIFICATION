import os
import shutil
import random
from collections import defaultdict

def split_dataset_by_video(input_base, output_base, train_ratio=0.7):
    labels = ['REAL', 'FAKE']
    splits = ['train', 'val']  # If you want to read all data, combine train+val first

    # Collect all files across train and val first
    all_files = {label: [] for label in labels}

    for split in splits:
        for label in labels:
            label_folder = os.path.join(input_base, split, label)
            if not os.path.exists(label_folder):
                continue
            for filename in os.listdir(label_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(label_folder, filename)
                    all_files[label].append(filepath)

    # Now group by video ID and split
    for label in labels:
        video_groups = defaultdict(list)
        for filepath in all_files[label]:
            filename = os.path.basename(filepath)
            video_id = filename.split('_scene')[0]
            video_groups[video_id].append(filepath)

        video_ids = list(video_groups.keys())
        random.shuffle(video_ids)

        split_point = int(len(video_ids) * train_ratio)
        train_ids = set(video_ids[:split_point])
        val_ids = set(video_ids[split_point:])

        # Prepare output folders
        for split_name in ['train', 'val']:
            output_folder = os.path.join(output_base, split_name, label)
            os.makedirs(output_folder, exist_ok=True)

        # Copy files according to split
        for video_id in train_ids:
            for file in video_groups[video_id]:
                dst = os.path.join(output_base, 'train', label, os.path.basename(file))
                shutil.copy2(file, dst)

        for video_id in val_ids:
            for file in video_groups[video_id]:
                dst = os.path.join(output_base, 'val', label, os.path.basename(file))
                shutil.copy2(file, dst)

        print(f"[{label}] ➤ {len(train_ids)} videos for training, {len(val_ids)} for validation.")

# Paths
input_folder = r'D:\HUMANS\SPLIT_DATASET'
output_folder = r'D:\HUMANS\TRAINING_AND_VALIDATION'

# Run the split
split_dataset_by_video(input_folder, output_folder, train_ratio=0.7)
