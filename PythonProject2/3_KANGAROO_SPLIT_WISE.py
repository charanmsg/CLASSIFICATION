import os
import shutil
import random
from collections import defaultdict

def safe_split_by_video(input_base, output_base, split_ratio=0.8):
    labels = ['REAL', 'FAKE']
    for label in labels:
        label_path = os.path.join(input_base, label)

        # Group frames by video
        video_groups = defaultdict(list)
        for filename in os.listdir(label_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                video_id = filename.split('_scene')[0]
                video_groups[video_id].append(filename)

        # Split videos
        video_ids = list(video_groups.keys())
        random.shuffle(video_ids)
        split_point = int(split_ratio * len(video_ids))
        train_ids = set(video_ids[:split_point])
        val_ids = set(video_ids[split_point:])

        for split_type, selected_ids in [('train', train_ids), ('val', val_ids)]:
            dest_dir = os.path.join(output_base, split_type, label)
            os.makedirs(dest_dir, exist_ok=True)

            for video_id in selected_ids:
                for file in video_groups[video_id]:
                    src = os.path.join(label_path, file)
                    dst = os.path.join(dest_dir, file)
                    shutil.copy2(src, dst)

        print(f"{label}: {len(train_ids)} videos for training, {len(val_ids)} for validation.")

# Example usage
input_base = r'D:\HUMANS\DE_DUPLICATION'
output_base = r'D:\HUMANS\SPLIT_DATASET'

safe_split_by_video(input_base, output_base)
