import cv2
import numpy as np
import imageio
import sys
import argparse
import multiprocessing
from functools import partial


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create a seamlessly looping GIF from a video."
    )
    parser.add_argument("--video", default="biboo.mp4", help="Input video file path")
    parser.add_argument("--output", default="output.gif", help="Output GIF file path")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=5.0,
        help="Maximum loop duration in seconds",
    )
    parser.add_argument(
        "--step", type=int, default=1, help="Frame step size for analysis"
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Downsample factor for frame comparison",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Output GIF frames per second"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help='Threshold for "perfect" loop detection (lower is better)',
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU processing even if GPU is available",
    )
    return parser.parse_args()


def frame_similarity(frame1, frame2, use_gpu=False):
    """Compare frames using histogram similarity with optional GPU acceleration"""
    if use_gpu and hasattr(cv2, "cuda"):
        try:
            # Upload frames to GPU
            gpu_frame1 = cv2.cuda_GpuMat()
            gpu_frame2 = cv2.cuda_GpuMat()
            gpu_frame1.upload(frame1)
            gpu_frame2.upload(frame2)

            # Convert to HSV in GPU
            gpu_hsv1 = cv2.cuda.cvtColor(gpu_frame1, cv2.COLOR_BGR2HSV)
            gpu_hsv2 = cv2.cuda.cvtColor(gpu_frame2, cv2.COLOR_BGR2HSV)

            # Download for histogram calculation (histograms not directly supported in CUDA)
            hsv1 = gpu_hsv1.download()
            hsv2 = gpu_hsv2.download()
        except Exception as e:
            print(f"GPU processing failed: {e}. Falling back to CPU.")
            # Fall back to CPU
            hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    else:
        # Use CPU
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    # Calculate histograms (CPU operation)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # Normalize histograms
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # Compare using correlation (higher is better, convert to distance)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return 1.0 - similarity  # Convert to distance (lower is better)


def compare_frames_batch(
    start_idx, frames_buffer, max_loop_frames, step, valid_frames, use_gpu=False
):
    """Compare one start frame with multiple end frames"""
    start_frame = frames_buffer[start_idx]
    local_min_diff = float("inf")
    local_best_end = 0

    # Require minimum loop length (at least 0.5 second worth of frames)
    min_frames = 15 // step

    for end_idx in range(
        start_idx + min_frames, min(start_idx + max_loop_frames // step, valid_frames)
    ):
        end_frame = frames_buffer[end_idx]

        # Check smoothness of transition - compare end frame with start frame
        transition_diff = frame_similarity(end_frame, start_frame, use_gpu)

        # Also check if frame before end is similar to frame after start
        if end_idx > 0 and start_idx < len(frames_buffer) - 1:
            before_end_frame = frames_buffer[end_idx - 1]
            after_start_frame = frames_buffer[start_idx + 1]
            transition_diff2 = frame_similarity(
                before_end_frame, after_start_frame, use_gpu
            )
            # Average the two transition scores
            transition_diff = (transition_diff + transition_diff2) / 2

        if transition_diff < local_min_diff:
            local_min_diff = transition_diff
            local_best_end = end_idx * step

    return (start_idx * step, local_best_end, local_min_diff)


def find_best_loop_segment(
    video_path,
    max_loop_duration=5.0,
    step=3,
    downsample_factor=4,
    perfect_threshold=0.01,
    use_gpu=False,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return None, None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_loop_frames = int(max_loop_duration * fps)

    # Store all frames within the analysis window
    frames_buffer = []
    print("Loading and processing frames...")
    frame_count = 0

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:  # Only keep frames according to step
            # Resize for faster comparison - use GPU if available
            if use_gpu and hasattr(cv2, "cuda"):
                try:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    gpu_small_frame = cv2.cuda.resize(
                        gpu_frame,
                        (
                            frame.shape[1] // downsample_factor,
                            frame.shape[0] // downsample_factor,
                        ),
                    )
                    small_frame = gpu_small_frame.download()
                except Exception as e:
                    # Fall back to CPU if GPU fails
                    print(f"GPU resize failed: {e}. Using CPU.")
                    small_frame = cv2.resize(
                        frame,
                        (
                            frame.shape[1] // downsample_factor,
                            frame.shape[0] // downsample_factor,
                        ),
                    )
            else:
                small_frame = cv2.resize(
                    frame,
                    (
                        frame.shape[1] // downsample_factor,
                        frame.shape[0] // downsample_factor,
                    ),
                )

            frames_buffer.append(small_frame)

        frame_count += 1

        if frame_count % 100 == 0:
            sys.stdout.write(f"\rLoaded {frame_count}/{total_frames} frames")
            sys.stdout.flush()

    print("\nFinding best loop segment...")
    valid_frames = len(frames_buffer)

    # Use multiprocessing to speed up comparisons
    cpu_count = multiprocessing.cpu_count()
    print(f"Using {cpu_count} CPU cores for processing")

    with multiprocessing.Pool(processes=cpu_count) as pool:
        compare_func = partial(
            compare_frames_batch,
            frames_buffer=frames_buffer,
            max_loop_frames=max_loop_frames,
            step=step,
            valid_frames=valid_frames,
            use_gpu=use_gpu,
        )

        # Create indices for all start frames to process
        start_indices = range(0, valid_frames - max_loop_frames // step)

        # Process in batches to show progress
        batch_size = max(1, len(start_indices) // 20)  # Show progress in ~20 updates
        results = []

        for i in range(0, len(start_indices), batch_size):
            batch_indices = list(start_indices[i : i + batch_size])
            batch_results = list(pool.map(compare_func, batch_indices))

            # Check if we found a "perfect" loop in this batch
            for result in batch_results:
                if result[2] < perfect_threshold:  # If difference is below threshold
                    print(f"\nFound perfect loop! Difference: {result[2]}")
                    return result[0], result[1]  # Early return with perfect match

            results.extend(batch_results)
            sys.stdout.write(
                f"\rProcessed {i+len(batch_indices)}/{len(start_indices)} start frames"
            )
            sys.stdout.flush()

    # Find best result
    best_start, best_end, min_diff = min(results, key=lambda x: x[2])

    print(
        f"\nBest loop segment found: frames {best_start} to {best_end}, difference: {min_diff}"
    )
    cap.release()
    return best_start, best_end


def extract_and_save_gif(
    video_path, start_frame, end_frame, output_gif, fps=12, step=1
):
    cap = cv2.VideoCapture(video_path)

    # More efficient extraction by reading frames sequentially
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []

    print("Extracting frames for GIF...")
    total = end_frame - start_frame

    # Extract frames using the same step as used during analysis
    for i in range(start_frame, end_frame, step):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for imageio
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        # Skip the frames in between (to match analysis step)
        for _ in range(step - 1):
            ret = cap.grab()  # Just grab and skip these frames
            if not ret:
                break

        if (i - start_frame) % (10 * step) == 0:
            sys.stdout.write(
                f"\rExtracting frame: {(i-start_frame)//step+1}/{total//step}"
            )
            sys.stdout.flush()

    # Create a multi-frame transition for smoother looping
    if len(frames) > 4:  # Need enough frames for a proper transition
        print("\nCreating multi-frame smooth transition...")
        
        # Create a series of transition frames between end and beginning
        num_transition_frames = min(3, len(frames) // 10)  # Up to 3 transition frames
        transition_frames = []
        
        for i in range(num_transition_frames):
            # Create a gradual transition
            alpha = (i + 1) / (num_transition_frames + 1)
            transition_frame = cv2.addWeighted(
                frames[-num_transition_frames + i], 1 - alpha,
                frames[i], alpha, 0
            )
            transition_frames.append(transition_frame)
        
        # Replace the last few frames with the transition frames
        frames = frames[:-num_transition_frames] + transition_frames

    cap.release()
    print("\nSaving GIF...")

    # Use a duration-based approach for more consistent timing
    # This helps eliminate stutters by ensuring exact frame timing
    durations = [1000.0/fps] * len(frames)  # Duration in ms for each frame
    
    # Make the last frame slightly longer to help viewers perceive the loop point
    if len(frames) > 0:
        durations[-1] = 1000.0/fps * 1.02  # Just 2% longer, barely noticeable
    
    # Ensure the GIF loops
    imageio.mimsave(output_gif, frames, duration=durations, loop=0)
    print(f"GIF saved as {output_gif}")


def check_gpu_support():
    """Check if OpenCV was built with CUDA support for GPU acceleration"""
    try:
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_devices > 0:
            print(f"GPU acceleration available: {cuda_devices} CUDA device(s) detected")
            return True
        else:
            print("No CUDA devices detected. Using CPU only.")
            return False
    except:
        print("OpenCV not built with CUDA support. Using CPU only.")
        return False


if __name__ == "__main__":
    args = parse_arguments()

    # Check for GPU support - use by default unless explicitly disabled
    use_gpu = not args.cpu_only and check_gpu_support()

    if use_gpu:
        print("Using GPU acceleration")
    else:
        if not args.cpu_only:
            print("GPU acceleration not available, using CPU")
        else:
            print("GPU acceleration disabled by user, using CPU")

    print(f"Processing video: {args.video}")
    print(f"Maximum loop duration: {args.max_duration}s")
    print(f"Using step size: {args.step} frames")
    print(f"Downsample factor: {args.downsample}")

    start, end = find_best_loop_segment(
        args.video,
        max_loop_duration=args.max_duration,
        step=args.step,
        downsample_factor=args.downsample,
        perfect_threshold=args.threshold,
        use_gpu=use_gpu,
    )

    # Add these lines to create the GIF
    if start is not None and end is not None:
        extract_and_save_gif(
            args.video, 
            start, 
            end, 
            args.output, 
            fps=args.fps, 
            step=args.step
        )
    else:
        print("Failed to find a seamless loop.")
