"""loopgif - turn a video with repeating motion into a perfectly looping GIF.

How it works:
  1. Decode every frame (tiny + grayscale) with ffmpeg.
  2. Each frame's tiny image is its "fingerprint".
  3. Compare all fingerprints against each other and find the pair of frames
     (start, end) that match best, i.e. where the motion returns to where it began.
  4. Keep frames start .. end-1 (the matching end frame is a duplicate of the
     start frame, so it is dropped to avoid a stutter on restart).
  5. Render that section as a GIF with a per-clip optimized palette.

Usage:
  python loopgif.py "video.mkv"
  python loopgif.py "video.mkv" -o out.gif --width 480 --fps 30
  python loopgif.py "video.mkv" --start 120 --end 160      (manual frames)
  python loopgif.py "video.mkv" --top 5                    (just list candidates)
  python loopgif.py "video.mkv" --rank 2                   (use the 2nd best match)

Requires: ffmpeg + ffprobe on PATH, numpy.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

FP_W, FP_H = 64, 36  # fingerprint size


def probe(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,width,height",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return float(num) / float(den), s["width"], s["height"]


def fingerprints(video):
    """Return an (N, FP_H*FP_W) float32 array, one row per frame."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-an",
           "-vf", f"scale={FP_W}:{FP_H}:flags=area,format=gray",
           "-vsync", "passthrough", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, FP_H * FP_W)
    return frames.astype(np.float32)


def distance_matrix(fp):
    """Mean absolute pixel difference between every pair of frames (0-255)."""
    n = len(fp)
    d = np.empty((n, n), np.float32)
    for i in range(n):  # row by row keeps memory small
        d[i] = np.abs(fp - fp[i]).mean(axis=1)
    return d


def find_loops(fp, fps, min_sec, max_sec, top):
    n = len(fp)
    d = distance_matrix(fp)
    motion = np.abs(np.diff(fp, axis=0)).mean(axis=1)  # change between consecutive frames
    cum_motion = np.concatenate([[0], np.cumsum(motion)])
    typical_motion = np.median(motion) + 1e-6

    min_len = max(2, int(round(min_sec * fps)))
    max_len = min(n - 1, int(round(max_sec * fps)))

    # Seam score: compare a few frames around the cut so the *movement* also
    # lines up, not just a single still pose.
    w = 2
    cands = []
    for i in range(w, n - w):
        for j in range(i + min_len, min(i + max_len, n - w - 1) + 1):
            seam = d[i - w:i + w + 1, j - w:j + w + 1].diagonal().mean()
            # Reject "loops" that are just a static shot (nothing moves).
            seg_motion = (cum_motion[j] - cum_motion[i]) / (j - i)
            if seg_motion < 0.3 * typical_motion:
                continue
            cands.append((seam, i, j))
    if not cands:
        return [], d
    cands.sort()
    best = cands[0][0]
    # Among near-best seams, prefer the shortest (one cycle, not two).
    tol = best * 1.15 + 0.15
    good = [c for c in cands if c[0] <= tol]
    good.sort(key=lambda c: (c[2] - c[1], c[0]))

    picked = []
    for c in good + cands:
        if any(abs(c[1] - p[1]) < 3 and abs(c[2] - p[2]) < 3 for p in picked):
            continue
        picked.append(c)
        if len(picked) >= top:
            break
    return picked, d


def render_gif(video, start, end, out, fps, width, src_fps):
    """Render frames [start, end) (end excluded) to a looping GIF."""
    vf =(f"select='between(n\\,{start}\\,{end - 1})',setpts=N/({src_fps}*TB),"
          f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors=256:stats_mode=full[p];"
          f"[b][p]paletteuse=dither=sierra2_4a")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-an",
           "-filter_complex", vf, "-loop", "0", str(out)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Make a perfectly looping GIF from a video.")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", help="output GIF (default: <video>_loop.gif)")
    ap.add_argument("--width", type=int, default=480, help="GIF width in px (default 480)")
    ap.add_argument("--fps", type=float, help="GIF frame rate (default: video's, capped at 30)")
    ap.add_argument("--min-sec", type=float, default=0.4, help="shortest loop to accept")
    ap.add_argument("--max-sec", type=float, default=6.0, help="longest loop to accept")
    ap.add_argument("--max-diff", type=float, default=6.0,
                    help="seam score above this counts as 'no loop found' (0-255 scale)")
    ap.add_argument("--start", type=int, help="manual start frame")
    ap.add_argument("--end", type=int, help="manual end frame (the repeat of start; excluded)")
    ap.add_argument("--top", type=int, default=0, help="list N best candidates and exit")
    ap.add_argument("--rank", type=int, default=1,
                    help="use the Nth best candidate instead of the best (see --top)")
    args = ap.parse_args()

    video = Path(args.video)
    src_fps, vw, vh = probe(video)
    fps = args.fps or min(src_fps, 30.0)
    out = Path(args.out) if args.out else video.with_name(video.stem + "_loop.gif")

    if args.start is not None and args.end is not None:
        start, end = args.start, args.end
        print(f"Using manual frames {start} -> {end}")
    else:
        print(f"Fingerprinting {video.name} ({vw}x{vh} @ {src_fps:g} fps)...")
        fp = fingerprints(video)
        print(f"  {len(fp)} frames")
        cands, _ = find_loops(fp, src_fps, args.min_sec, args.max_sec, max(args.top, args.rank))
        if args.top:
            print(f"\nTop {len(cands)} loop candidates (lower score = smoother seam):")
            for s, i, j in cands:
                print(f"  frames {i:5d} -> {j:5d}  ({(j - i) / src_fps:5.2f}s, "
                      f"starts at {i / src_fps:6.2f}s)  score {s:.2f}")
            return
        if not cands or cands[0][0] > args.max_diff:
            msg = f" (best score {cands[0][0]:.2f})" if cands else ""
            sys.exit(f"Couldn't find a loop{msg}. Try --top 10 to inspect candidates, "
                     f"or pass --start/--end frames yourself.")
        if args.rank > len(cands):
            sys.exit(f"Only {len(cands)} candidate(s) found; --rank {args.rank} is out of range.")
        score, start, end = cands[args.rank - 1]
        print(f"Loop found: frames {start} -> {end} "
              f"({(end - start) / src_fps:.2f}s, starts at {start / src_fps:.2f}s), "
              f"seam score {score:.2f}")

    render_gif(video, start, end, out, fps, args.width, src_fps)
    size = out.stat().st_size / 1e6
    print(f"Saved {out} ({end - start} frames, {size:.1f} MB)")


if __name__ == "__main__":
    main()
