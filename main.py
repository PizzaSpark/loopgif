import cv2
import numpy as np
import argparse
from PIL import Image
import os
from tqdm import tqdm

def calculate_frame_similarity(frame1, frame2):
    """Calculate similarity between two frames using structural similarity"""
    # Convert to grayscale for simpler comparison
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate Mean Squared Error
    mse = np.mean((gray1 - gray2) ** 2)
    if mse == 0:  # Images are identical
        return 1.0
    
    # Calculate similarity (inverse of difference)
    similarity = 1.0 - (mse / 255**2)
    return similarity

def find_loop_in_video(video_path, similarity_threshold=0.95, min_loop_length=5, max_loop_length=None):
    """Find repeating animation loops in a video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video")
        return None
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Store frames for processing
    frames = []
    print("Reading video frames...")
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    # If max_loop_length not specified, set to full video length
    if max_loop_length is None:
        max_loop_length = len(frames) - 1
    
    best_loop = None
    highest_similarity = 0
    
    print("Searching for animation loops...")
    # Changed to use full video length instead of half
    for loop_length in tqdm(range(min_loop_length, min(max_loop_length, len(frames) - 1))):
        for start_frame in range(len(frames) - (2 * loop_length)):
            # Compare frames in potential loops
            loop1_frames = frames[start_frame:start_frame + loop_length]
            loop2_frames = frames[start_frame + loop_length:start_frame + (2 * loop_length)]
            
            # Calculate average similarity between corresponding frames
            similarities = [
                calculate_frame_similarity(loop1_frames[i], loop2_frames[i])
                for i in range(loop_length)
            ]
            avg_similarity = sum(similarities) / loop_length
            
            if avg_similarity > highest_similarity and avg_similarity >= similarity_threshold:
                highest_similarity = avg_similarity
                best_loop = {
                    "start_frame": start_frame,
                    "length": loop_length,
                    "similarity": avg_similarity,
                    "frames": loop1_frames
                }
    
    return best_loop

def create_gif_from_frames(frames, output_path, fps=24):
    """Create a GIF from a list of frames"""
    if not frames:
        print("Error: No frames provided")
        return False
    
    # Convert frames from BGR to RGB
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    
    # Calculate appropriate duration (in milliseconds)
    duration = int(1000 / fps)
    
    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=True,
        duration=duration,
        loop=0  # Loop indefinitely
    )
    print(f"GIF saved to {output_path}")
    return True

def main():
    video_path = "Biboo cute wiggles.mp4"
    output_path = "output.gif"
    similarity_threshold = 0.95            # Similarity threshold (0.0-1.0)
    min_loop_length = 5                    # Minimum loop length in frames
    max_loop_length = None                 # No maximum loop length (will use half video length)
    
    print(f"Processing video: {video_path}")
    best_loop = find_loop_in_video(
        video_path, 
        similarity_threshold=similarity_threshold,
        min_loop_length=min_loop_length,
        max_loop_length=max_loop_length
    )
    
    if best_loop:
        print(f"Found loop at frame {best_loop['start_frame']} with {best_loop['length']} frames")
        print(f"Loop similarity: {best_loop['similarity']:.4f}")
        create_gif_from_frames(best_loop['frames'], output_path)
    else:
        print("No suitable animation loop found in the video")

if __name__ == "__main__":
    main()