import os
import shutil
from PIL import Image
import imagehash

def deduplicate_and_save(input_base, output_base, hash_size=8, threshold=5):
    labels = ['REAL', 'FAKE']

    for label in labels:
        input_folder = os.path.join(input_base, label)
        output_folder = os.path.join(output_base, label)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        seen_hashes = {}
        kept_count = 0
        skipped_count = 0

        for filename in os.listdir(input_folder):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            input_path = os.path.join(input_folder, filename)

            try:
                with Image.open(input_path) as img:
                    img_hash = imagehash.phash(img, hash_size=hash_size)

                # Check for similarity with previously seen images
                is_duplicate = False
                for existing_hash in seen_hashes:
                    if abs(img_hash - existing_hash) <= threshold:
                        is_duplicate = True
                        skipped_count += 1
                        print(f"Duplicate skipped: {filename}")
                        break

                if not is_duplicate:
                    seen_hashes[img_hash] = filename
                    shutil.copy2(input_path, os.path.join(output_folder, filename))
                    kept_count += 1

            except Exception as e:
                print(f"Error with file {filename}: {e}")

        print(f"\n[{label}] Done: {kept_count} unique images kept, {skipped_count} duplicates skipped.")

# Define your paths
input_base_path = r'D:\HUMANS\FRAMES'
output_base_path = r'D:\HUMANS\DE_DUPLICATION'

deduplicate_and_save(input_base_path, output_base_path)
