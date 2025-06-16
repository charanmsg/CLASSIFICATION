import os
import cv2

def extract_all_frames_flat(base_path, output_path):
    labels = ['REAL', 'FAKE']

    for label in labels:
        video_folder = os.path.join(base_path, label)
        output_label_folder = os.path.join(output_path, label)

        os.makedirs(output_label_folder, exist_ok=True)

        for video_file in os.listdir(video_folder):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue  # Skip non-video files

            video_path = os.path.join(video_folder, video_file)
            video_name = os.path.splitext(video_file)[0]

            cap = cv2.VideoCapture(video_path)
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_filename = f"{video_name}_frame_{count:04d}.jpg"
                frame_path = os.path.join(output_label_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
                count += 1

            cap.release()
            print(f"Extracted {count} frames from {video_file} into {output_label_folder}")

# Example usage
base_video_path = r'D:\HUMANS'
frames_output_path = r'D:\HUMANS\FRAMES'

extract_all_frames_flat(base_video_path, frames_output_path)
