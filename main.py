import cv2
import numpy as np
import imagehash
from PIL import Image
import os
import argparse
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("Extracting frames...")
    with tqdm(total=total_frames, desc="Extracting frames") as progress_bar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(output_folder, f"frame_{frame_idx}.png")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            frame_idx += 1
            progress_bar.update(1)
    
    cap.release()
    print(f"Total frames extracted: {len(frames)}")
    return frames

def compute_ssim(frame1, frame2):
    img1 = cv2.imread(frame1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(frame2, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (100, 100))  # Resize to speed up processing
    img2 = cv2.resize(img2, (100, 100))
    return ssim(img1, img2)

def find_matching_frames(frames, threshold=0.95):
    print("Finding matching frames...")
    total_comparisons = sum(1 for i in range(len(frames)) for j in range(i + 30, len(frames)))
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor, tqdm(total=total_comparisons, desc="Comparing frames") as progress_bar:
        future_to_pair = {}
        futures = []
        
        for i in range(len(frames)):
            for j in range(i + 30, len(frames)):
                future = executor.submit(compute_ssim, frames[i], frames[j])
                future_to_pair[future] = (i, j)
                futures.append(future)
        
        for future in as_completed(futures):
            similarity = future.result()
            progress_bar.update(1)
            if similarity > threshold:
                matching_frames = future_to_pair[future]
                print(f"Found matching frames: {matching_frames}")
                
                # Cancel remaining tasks
                executor.shutdown(wait=False, cancel_futures=True)
                return matching_frames
    
    print("No suitable matching frames found.")
    return None, None

def create_gif(frames, start, end, output_gif):
    print(f"Creating GIF from frame {start} to frame {end}...")
    images = [Image.open(frames[i]) for i in range(start, end)]
    images[0].save(output_gif, save_all=True, append_images=images[1:], loop=0, duration=50)
    print(f"GIF saved as {output_gif}")

def main():
    parser = argparse.ArgumentParser(description="Create a looping GIF from a video.")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("output", help="Path to the output GIF file")
    parser.add_argument("--threshold", type=float, default=0.95, help="SSIM similarity threshold (higher is stricter)")
    
    args = parser.parse_args()
    temp_folder = "temp_frames"
    frames = extract_frames(args.video, temp_folder)
    start, end = find_matching_frames(frames, args.threshold)
    
    if start is not None and end is not None:
        create_gif(frames, start, end, args.output)
    else:
        print("No suitable looping frames found. Try adjusting the threshold.")
    
    print("Cleaning up temporary files...")
    for frame in frames:
        os.remove(frame)
    os.rmdir(temp_folder)
    print("Done!")

if __name__ == "__main__":
    main()