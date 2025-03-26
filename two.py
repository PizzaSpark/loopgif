import os
import cv2
import numpy as np
from PIL import Image
from itertools import combinations
from multiprocessing import Pool, cpu_count

def resize_keep_aspect_ratio(image, target_height=128):
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)

def extract_frames(video_path, output_folder, target_height=128):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()
    
    while success:
        resized_frame = resize_keep_aspect_ratio(frame, target_height)
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, resized_frame)
        success, frame = cap.read()
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} downscaled frames.")

def dhash(image, hash_size=8):
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def load_image_hashes(folder_path):
    frame_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.jpg')],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    frame_hashes = {}
    
    for frame in frame_files:
        img = cv2.imread(os.path.join(folder_path, frame), cv2.IMREAD_GRAYSCALE)
        frame_hashes[frame] = dhash(img)
    
    return frame_files, frame_hashes

def compare_pair(pair):
    frame1, frame2, hashes = pair
    diff = bin(hashes[frame1] ^ hashes[frame2]).count("1")
    return frame1, frame2, diff

def find_closest_matching_frames(folder_path, threshold=1, checked_pairs=set()):
    frame_files, frame_hashes = load_image_hashes(folder_path)
    pairs = [(frame1, frame2, frame_hashes) for frame1, frame2 in combinations(frame_files, 2) if (frame1, frame2) not in checked_pairs]
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(compare_pair, pairs)
    
    for frame1, frame2, diff in results:
        checked_pairs.add((frame1, frame2))
        if diff <= threshold:
            frame1_num, frame2_num = int(frame1.split("_")[1].split(".")[0]), int(frame2.split("_")[1].split(".")[0])
            print(f"✅ Found similar frames: {frame1_num} and {frame2_num} (Difference: {diff})")
            return frame1_num, frame2_num, checked_pairs
    
    print("❌ No similar frames found.")
    return None, None, checked_pairs

def find_alternative_matching_frame(frame_start, folder_path, used_frames, threshold=1, checked_pairs=set()):
    frame_files, frame_hashes = load_image_hashes(folder_path)
    for frame in frame_files:
        frame_num = int(frame.split("_")[1].split(".")[0])
        if frame_num != frame_start and frame_num not in used_frames and (f"frame_{frame_start}.jpg", frame) not in checked_pairs:
            diff = bin(frame_hashes[f"frame_{frame_start}.jpg"] ^ frame_hashes[frame]).count("1")
            checked_pairs.add((f"frame_{frame_start}.jpg", frame))
            if diff <= threshold:
                return frame_num, checked_pairs
    return None, checked_pairs

def extract_original_frames(video_path, output_folder, start_frame, end_frame):
    if start_frame is None or end_frame is None:
        return
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()
    
    while success:
        if start_frame <= frame_count < end_frame:
            frame_filename = os.path.join(output_folder, f"original_frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
        
        success, frame = cap.read()
        frame_count += 1
    
    cap.release()
    print(f"Extracted original frames from {start_frame} to {end_frame-1}.")

def create_gif(frames, start, end, output_gif, video_path):
    print(f"Creating GIF from frame {start} to frame {end-1}...")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        fps = 30
    frame_duration = int(1000 / fps)

    images = [Image.open(frames[i]) for i in range(len(frames)) if start <= int(os.path.basename(frames[i]).split("_")[2].split(".")[0]) < end]
    images[0].save(output_gif, save_all=True, append_images=images[1:], loop=0, duration=frame_duration)

    print(f"GIF saved as {output_gif} (Duration per frame: {frame_duration} ms)")

def cleanup_folders(*folders):
    for folder in folders:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                os.remove(file_path)
            os.rmdir(folder)
    print("✅ Cleaned up temporary files and folders.")

if __name__ == "__main__":
    video_path = "input.mp4"
    downscaled_folder = "downscaled_frames"
    original_folder = "original_frames"
    gif_output = "output.gif"
    
    extract_frames(video_path, downscaled_folder)
    checked_pairs = set()
    
    while True:
        frame_start, frame_end, checked_pairs = find_closest_matching_frames(downscaled_folder, checked_pairs=checked_pairs)
        if frame_start is None:
            print("No alternative matching frames found.")
            break
        
        extract_original_frames(video_path, original_folder, frame_start, frame_end)
        frame_files = sorted([os.path.join(original_folder, f) for f in os.listdir(original_folder) if f.endswith('.jpg')],
                             key=lambda x: int(os.path.basename(x).split("_")[2].split(".")[0]))
        
        create_gif(frame_files, frame_start, frame_end, gif_output, video_path)
        
        user_input = input("Are you satisfied with the GIF? (y/n): ")
        if user_input.lower() == 'y':
            break
    
    cleanup_folders(downscaled_folder, original_folder)
