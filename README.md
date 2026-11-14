# loopgif

Turn a video clip with repeating motion into a GIF that loops perfectly.

Give it a clip of something that repeats, like an anime character swaying side to side,
and it finds where the motion starts over and cuts out exactly one cycle. The result
loops forever with no jump when it restarts.

## How it works

Think of a flipbook. If page 5 looks exactly like page 25, then pages 5–24 are one
full cycle of the motion.

1. Every frame is shrunk to a tiny 64×36 grayscale image. That's its **fingerprint**.
2. Every fingerprint is compared with every other one that's a possible loop length
   away (`--min-sec` to `--max-sec`).
3. A possible cut is scored by comparing a few frames on *both sides* of it, so the
   movement has to line up, not just one still pose.
4. Sections that barely move (`--min-motion`) are ignored so a still shot isn't
   mistaken for a loop. This works for anime too, where most frames are held.
5. If two cycle lengths score about the same, the shorter one wins (one cycle, not two).
6. The section is rendered as a GIF with a palette tuned to that clip. The matching end
   frame is dropped because it duplicates the first frame and would cause a stutter.
   If the GIF frame rate is lower than the video's, frames are spaced evenly across
   the loop so the seam stays smooth.

A 20-second 1080p clip takes about 4 seconds; a whole 4-minute music video about 13.
Most of that is ffmpeg decoding the video.

## Requirements

- Python 3.9+
- A recent [ffmpeg](https://ffmpeg.org/) (tested with 9.0), with ffmpeg and ffprobe
  on your PATH. On Windows: `scoop install ffmpeg` or `winget install Gyan.FFmpeg`.
  The script uses `-fps_mode`, which replaced `-vsync`, so very old builds without
  `-fps_mode` won't work (and newer ones, e.g. 9.0, no longer accept `-vsync`).
- numpy (`pip install -r requirements.txt`)

## Usage

```bash
python loopgif.py "clip.mkv"                         # find the loop, write clip_loop.gif
python loopgif.py "clip.mkv" -o out.gif --width 360  # smaller output
python loopgif.py "clip.mkv" --top 10                # list the best cut points
python loopgif.py "clip.mkv" --rank 2                # don't like the result? use the 2nd best
python loopgif.py "clip.mkv" --from 0:05 --to 0:12   # only search that part of the video
python loopgif.py "clip.mkv" --start 120 --end 169   # choose the frames yourself
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-o, --out` | `<video>_loop.gif` | Output file |
| `--width` | 480 | GIF width in px (height keeps aspect) |
| `--fps` | video fps, max 30 | GIF frame rate (lower = smaller file) |
| `--from` / `--to` | whole video | Only search this time range (`12.5`, `1:05`, `1:02:03`). Frame numbers shown still count from the start of the video |
| `--min-sec` / `--max-sec` | 0.4 / 6.0 | Allowed loop length |
| `--max-diff` | 6.0 | Seam score above this counts as "no loop found" |
| `--min-motion` | 0.5 | Skip sections that move less than this per frame (filters out still shots) |
| `--top N` | – | Print the N best candidates and exit. The same loop shifted by a few frames is only listed once |
| `--rank N` | 1 | Use the Nth best candidate |
| `--start` / `--end` | – | Manual frames (give both); `end` is the repeat of `start` and is left out |

## Limits

It only works if the clip actually repeats. If the motion never comes back to where it
started, it says it couldn't find a loop. Use `--top` to look at the near misses, or pick
the frames yourself with `--start`/`--end`.

## History

This repo combines two earlier experiments, kept in [`legacy/`](legacy):

- `legacy/one/` – from *auto-gif-maker-python*: OpenCV HSV-histogram matching with
  multiprocessing and optional CUDA.
- `legacy/two/` – from *auto-gif-maker-two-python*: SSIM matching (`main.py`) and a
  dhash version with a "try the next match" prompt (`two.py`).

`loopgif.py` replaces both. It doesn't need OpenCV and it's faster. The "try the next
match" idea lives on as `--rank`.
