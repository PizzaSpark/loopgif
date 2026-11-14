"""loopgif - turn a video with repeating motion into a perfectly looping GIF.

How it works:
  1. Decode every frame (tiny + grayscale) with ffmpeg.
  2. Each frame's tiny image is its "fingerprint".
  3. Compare fingerprints that are a plausible loop length apart and find the
     pair of frames (start, end) that match best, i.e. where the motion returns
     to where it began.
  4. Keep frames start .. end-1 (the matching end frame is a duplicate of the
     start frame, so it is dropped to avoid a stutter on restart).
  5. Render that section as a GIF with a per-clip optimized palette.

Usage:
  python loopgif.py "video.mkv"
  python loopgif.py "video.mkv" -o out.gif --width 480 --fps 30
  python loopgif.py "video.mkv" --start 120 --end 160      (manual frames)
  python loopgif.py "video.mkv" --top 5                    (just list candidates)
  python loopgif.py "video.mkv" --rank 2                   (use the 2nd best match)
  python loopgif.py "video.mkv" --from 0:05 --to 0:12      (only search that part)

Requires: a recent ffmpeg with -fps_mode (ffmpeg + ffprobe on PATH), numpy.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

FP_W, FP_H = 64, 36  # fingerprint size
SEAM_W = 2           # frames on each side of the cut that must also line up


def run_ff(cmd, what):
    """Run an ffmpeg/ffprobe command; on failure exit with ffmpeg's own message."""
    try:
        return subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode("utf-8", "replace").strip() or f"exit status {e.returncode}"
        sys.exit(f"error: {cmd[0]} failed while {what}:\n{msg}")


def probe(video):
    out = run_ff(["ffprobe", "-v", "error", "-select_streams", "v:0",
                  "-show_entries", "stream=r_frame_rate,avg_frame_rate,width,height",
                  "-of", "json", str(video)], "reading the video info")
    streams = json.loads(out).get("streams") or []
    if not streams:
        sys.exit(f"error: no video stream in {video}")
    s = streams[0]
    for key in ("r_frame_rate", "avg_frame_rate"):
        num, _, den = s.get(key, "0/0").partition("/")
        if float(num or 0) > 0 and float(den or 1) > 0:
            return float(num) / float(den or 1), s["width"], s["height"]
    sys.exit(f"error: couldn't read the frame rate of {video}")


def parse_time(s):
    """'12.5' -> 12.5, '1:05' -> 65.0, '1:02:03.5' -> 3723.5"""
    secs = 0.0
    for part in s.split(":"):
        secs = secs * 60 + float(part)
    return secs


def range_args(t_from, t_to):
    """ffmpeg input options that limit decoding to [t_from, t_to)."""
    args = []
    if t_from:
        args += ["-ss", f"{t_from:.6f}"]
    if t_to is not None:
        args += ["-t", f"{t_to - (t_from or 0):.6f}"]
    return args


def fingerprints(video, rng=()):
    """Return an (N, FP_H*FP_W) uint8 array, one row per frame."""
    cmd = ["ffmpeg", "-v", "error", *rng, "-i", str(video), "-an",
           "-vf", f"scale={FP_W}:{FP_H}:flags=area,format=gray",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    raw = run_ff(cmd, "decoding the video")
    return np.frombuffer(raw, np.uint8).reshape(-1, FP_H * FP_W)


def lag_diff(fp, lag, chunk=4096):
    """out[t] = mean absolute pixel difference between frame t and t+lag (0-255)."""
    n = len(fp) - lag
    out = np.empty(max(n, 0), np.float32)
    for s in range(0, n, chunk):  # chunked so long videos don't need huge temporaries
        e = min(s + chunk, n)
        a = fp[s:e].astype(np.int16)
        b = fp[s + lag:e + lag].astype(np.int16)
        out[s:e] = np.abs(a - b).mean(axis=1)
    return out


def find_loops(fp, fps, min_sec, max_sec, min_motion, top):
    """Return up to `top` distinct (score, start, end) loops, best first."""
    n = len(fp)
    w = SEAM_W
    motion = lag_diff(fp, 1)  # change between consecutive frames
    cum_motion = np.concatenate([[0.0], np.cumsum(motion, dtype=np.float64)])
    # A loop must move at least this much per frame on average, otherwise it's
    # a still shot. An absolute floor (not relative to the video's median) so
    # it works for held-frame anime, where most frame steps are ~0. Measured:
    # still shots ~0.02, real loops 2+.
    motion_floor = min_motion

    min_len = max(2, int(round(min_sec * fps)))
    max_len = int(round(max_sec * fps))

    scores, starts, lens = [], [], []
    for L in range(min_len, max_len + 1):
        last_i = n - 1 - w - L  # j + w must stay inside the video
        if last_i < w:
            break
        d = lag_diff(fp, L)
        cs = np.concatenate([[0.0], np.cumsum(d, dtype=np.float64)])
        i = np.arange(w, last_i + 1)
        # Seam score: average match of frames i-w..i+w against j-w..j+w, so the
        # *movement* through the cut lines up, not just a single still pose.
        seam = (cs[i + w + 1] - cs[i - w]) / (2 * w + 1)
        seg_motion = (cum_motion[i + L] - cum_motion[i]) / L
        keep = seg_motion >= motion_floor
        scores.append(seam[keep])
        starts.append(i[keep])
        lens.append(np.full(int(keep.sum()), L))
    if not scores or not sum(len(s) for s in scores):
        return []
    scores = np.concatenate(scores)
    starts = np.concatenate(starts)
    lens = np.concatenate(lens)

    by_score = np.argsort(scores, kind="stable")
    best = scores[by_score[0]]
    # Among near-best seams, prefer the shortest (one cycle, not two).
    tol = best * 1.15 + 0.15
    good = by_score[scores[by_score] <= tol]
    good = good[np.lexsort((scores[good], lens[good]))]

    picked = []
    for k in np.concatenate([good, by_score]):
        s, i, L = float(scores[k]), int(starts[k]), int(lens[k])
        # Same cycle length starting within one cycle of a picked loop = the
        # same loop, just rotated. Skip it so --top/--rank show real choices.
        if any(abs(L - pL) <= max(2, pL // 10) and abs(i - pi) < pL
               for _, pi, pj in picked for pL in [pj - pi]):
            continue
        picked.append((s, i, i + L))
        if len(picked) >= top:
            break
    return picked


def render_gif(video, start, end, out, fps, width, src_fps, rng=()):
    """Render frames [start, end) (end excluded) to a looping GIF.
    Frame numbers count from the start of `rng` (or the video if no range).
    Returns the number of frames in the GIF."""
    n_src = end - start
    # Resample so the loop is a whole number of GIF frames, spaced evenly;
    # otherwise dropping frames (e.g. 60 -> 30 fps) can leave an uneven seam.
    n_out = max(1, round(n_src * fps / src_fps))
    out_fps = n_out * src_fps / n_src
    vf = (f"trim=start_frame={start}:end_frame={end},setpts=N/({src_fps}*TB),"
          f"fps={out_fps:.6f},scale={width}:-1:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors=256:stats_mode=full[p];"
          f"[b][p]paletteuse=dither=sierra2_4a")
    cmd = ["ffmpeg", "-v", "error", "-y", *rng, "-i", str(video), "-an",
           "-filter_complex", vf, "-loop", "0", str(out)]
    run_ff(cmd, "rendering the GIF")
    return n_out


def main():
    # Filenames with e.g. Japanese characters crash print() when output is
    # piped/redirected on Windows (cp1252); replace unprintable chars instead.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Make a perfectly looping GIF from a video.")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", help="output GIF (default: <video>_loop.gif)")
    ap.add_argument("--width", type=int, default=480, help="GIF width in px (default 480)")
    ap.add_argument("--fps", type=float, help="GIF frame rate (default: video's, capped at 30)")
    ap.add_argument("--min-sec", type=float, default=0.4, help="shortest loop to accept")
    ap.add_argument("--max-sec", type=float, default=6.0, help="longest loop to accept")
    ap.add_argument("--max-diff", type=float, default=6.0,
                    help="seam score above this counts as 'no loop found' (0-255 scale)")
    ap.add_argument("--min-motion", type=float, default=0.5,
                    help="skip sections that move less than this per frame "
                         "(0-255 scale; filters out still shots)")
    ap.add_argument("--from", dest="t_from", type=parse_time, metavar="TIME",
                    help="only search from this time (seconds or m:ss)")
    ap.add_argument("--to", dest="t_to", type=parse_time, metavar="TIME",
                    help="only search up to this time (seconds or m:ss)")
    ap.add_argument("--start", type=int, help="manual start frame")
    ap.add_argument("--end", type=int, help="manual end frame (the repeat of start; excluded)")
    ap.add_argument("--top", type=int, default=0, help="list N best candidates and exit")
    ap.add_argument("--rank", type=int, default=1,
                    help="use the Nth best candidate instead of the best (see --top)")
    args = ap.parse_args()

    if (args.start is None) != (args.end is None):
        ap.error("--start and --end must be given together.")
    if args.start is not None and not 0 <= args.start < args.end:
        ap.error("--end must be greater than --start (and --start at least 0).")
    if args.rank < 1:
        ap.error("--rank must be 1 or more.")
    if args.min_sec <= 0 or args.max_sec < args.min_sec:
        ap.error("need 0 < --min-sec <= --max-sec.")
    if args.t_from is not None and args.t_to is not None and args.t_to <= args.t_from:
        ap.error("--to must be later than --from.")

    missing = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if missing:
        sys.exit(f"error: {' and '.join(missing)} not found on PATH. "
                 "Install ffmpeg (e.g. `scoop install ffmpeg` or "
                 "`winget install Gyan.FFmpeg`) and open a new terminal.")

    video = Path(args.video)
    if not video.is_file():
        sys.exit(f"error: video not found: {video}")
    src_fps, vw, vh = probe(video)
    fps = args.fps or min(src_fps, 30.0)
    out = Path(args.out) if args.out else video.with_name(video.stem + "_loop.gif")

    # Frame numbers shown to the user always count from the start of the video.
    # Inside a --from/--to range, ffmpeg numbers frames from the range start, so
    # `off` converts between the two.
    rng, off = (), 0
    if args.start is not None:
        start, end = args.start, args.end
        print(f"Using manual frames {start} -> {end}")
    else:
        rng = range_args(args.t_from, args.t_to)
        off = int(np.ceil((args.t_from or 0) * src_fps - 1e-6))
        where = ""
        if rng:
            t_to = f"{args.t_to:g}s" if args.t_to is not None else "end"
            where = f", searching {args.t_from or 0:g}s - {t_to}"
        print(f"Fingerprinting {video.name} ({vw}x{vh} @ {src_fps:g} fps{where})...")
        fp = fingerprints(video, rng)
        print(f"  {len(fp)} frames")
        if len(fp) < 2:
            sys.exit("That range has no frames. Check --from/--to.")
        cands = find_loops(fp, src_fps, args.min_sec, args.max_sec, args.min_motion,
                           max(args.top, args.rank))
        if args.top:
            print(f"\nTop {len(cands)} loop candidates (lower score = smoother seam):")
            for s, i, j in cands:
                print(f"  frames {i + off:5d} -> {j + off:5d}  ({(j - i) / src_fps:5.2f}s, "
                      f"starts at {(i + off) / src_fps:6.2f}s)  score {s:.2f}")
            if not cands:
                print("  none (everything was too short, too long, or not moving)")
            return
        if not cands:
            sys.exit("Couldn't find a loop: no moving section of the right length. "
                     "Try a different --from/--to, --min-sec/--max-sec, or a lower --min-motion.")
        if args.rank > len(cands):
            sys.exit(f"Only {len(cands)} distinct candidate(s) found; "
                     f"--rank {args.rank} is out of range.")
        score, start, end = cands[args.rank - 1]
        if score > args.max_diff:
            sys.exit(f"Couldn't find a loop (candidate #{args.rank} scores {score:.2f}, "
                     f"above --max-diff {args.max_diff:g}). Try --top 10 to inspect "
                     f"candidates, or pass --start/--end frames yourself.")
        print(f"Loop found: frames {start + off} -> {end + off} "
              f"({(end - start) / src_fps:.2f}s, starts at {(start + off) / src_fps:.2f}s), "
              f"seam score {score:.2f}")

    n_frames = render_gif(video, start, end, out, fps, args.width, src_fps, rng)
    size = out.stat().st_size / 1e6
    print(f"Saved {out} ({n_frames} frames, {size:.1f} MB)")


if __name__ == "__main__":
    main()
