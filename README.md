# YOLO → OSC Multi-Object Controller (Tennis-Ball Ready)

Real-time multi-object tracker using Ultralytics YOLO (v8/v11…) and BYTETracker/BoT-SORT, outputting normalized **x, y, size** via **OSC** for up to **N slots**. Default class is **`sports ball`** (tennis ball works great), but you can track any COCO classes (e.g., `cup`, `bottle`, `banana`) without retraining.

## Features
- **Multi-target** tracking with persistent IDs (uses `track()` with `persist=True`).
- **Up to N slots** (default 3):  
  - `/ball/1 [x y size]`, `/ball/2 [...]`, `/ball/3 [...]`
- **Normalized outputs**:  
  - `x`, `y` in `[0..1]` (origin at top-left)  
  - `size = bbox_area / frame_area` (proxy for distance)
- **Robustness**:
  - EMA smoothing (`--ema`)
  - Dropout hold (`--hold`)
  - Minimum size gate (`--min-area`)
- **Flexible classes** via `--classes "sports ball,cup,banana"`
- **Fast on CPU**, optional GPU/CUDA or Apple `mps`

## Install
```bash
python -m venv .venv && source .venv/bin/activate   # (optional)
pip install ultralytics opencv-python python-osc
pip install --no-cache-dir "lap>=0.5.12"
```

> **Models:** This script expects a COCO-trained YOLO checkpoint, e.g. `yolov8n.pt`, `yolov8s.pt`, `yolo11n.pt`, `yolo11s.pt`. Place it where Ultralytics can download/load it automatically, or pass `--model /path/to/weights.pt`.

## Quick Start (tennis ball)
```bash
python sports_ball_multitracker.py \
  --classes "sports ball" --max-slots 3 --imgsz 320 --ema 0.25
```

### In Max/MSP (example)
Listen on the same port you pass with `--osc-port` (default `9000`):
```
[udpreceive 9000]
|
[route /ball/1 /ball/2 /ball/3 /balls/count]
```
Each `/ball/n` outputs `[x y size]`. Sentinel for “no target” is `[-1.0, -1.0, 0.0]`.

## Useful Flags
| Flag | Meaning | Tip |
|---|---|---|
| `--classes "sports ball,cup,banana"` | Which COCO classes to track | Default `"sports ball"` |
| `--max-slots 3` | How many tagged outputs (`/ball/1..N`) | 1–3 works well live |
| `--imgsz 256..384` | Inference resolution | Smaller = faster; 320 is a good CPU default |
| `--conf 0.20..0.35` | Detection confidence | Lower catches more, risks more noise |
| `--iou 0.45..0.55` | NMS IoU | Slight increases may keep nearby boxes |
| `--ema 0.0..0.6` | Smoothing | 0.25–0.4 feels stable |
| `--hold 8..30` | Frames to hold last value on brief misses | Increase for fast moves/occlusion |
| `--min-area 0.0005..0.01` | Reject tiny boxes | Raise to ignore distant noise |
| `--tracker bytetrack.yaml` | Tracker choice | `botsort.yaml` = stickier IDs; ByteTrack = faster |
| `--device cpu|cuda|mps|auto` | Compute backend | `auto` picks CUDA if available |
| `--no-video` | Headless mode | Good for stage machines |
| `--fps-cap 15` | Limit processing FPS | Stabilizes CPU usage |

### Profiles
**CPU-fast:**
```bash
python sports_ball_multitracker.py \
  --model yolov8n.pt --imgsz 288 --conf 0.20 --iou 0.50 --ema 0.3 \
  --tracker bytetrack.yaml --max-slots 3
```

**GPU / Apple M-series:**
```bash
# CUDA
python sports_ball_multitracker.py --model yolo11s.pt --device cuda --imgsz 384 --ema 0.25 --tracker botsort.yaml
# Apple Silicon (Metal)
python sports_ball_multitracker.py --model yolov8s.pt --device mps --imgsz 384 --ema 0.25 --tracker botsort.yaml
```

## How Slot Assignment Works
1. Keep existing **trackID → slot** mappings when possible.  
2. Fill empty slots with new tracks.  
3. If full, replace the slot whose current detection has the **smallest area** with a **larger** new detection.  
4. Per-slot EMA and hold are independent.

> Want a different policy (e.g., prefer most centered objects, or “lock” a slot until its target is gone for N frames)? You can adjust the assignment section easily.

## Troubleshooting
- **Low FPS on CPU**: reduce `--imgsz` to 288 or 256; add `--fps-cap 15`. Keep capture at 640×480 or 320×240.
- **Drops when moving fast**: lower `--conf` (e.g., 0.20), raise `--hold` (e.g., 20), or try `--tracker botsort.yaml`.
- **Wrong/extra detections**: raise `--conf`, raise `--min-area`, or restrict `--classes` to the specific object.
- **No detections**: improve lighting, move object closer, lower `--conf`, or increase `--imgsz` slightly (e.g., 384).

## OSC Contract
- **Per slot:** `/ball/<n> [x y size]`  
  - Sentinel (no target): `[-1.0, -1.0, 0.0]`
- **Active count:** `/balls/count n_active`

All values are floats. `x, y` are normalized to the current camera frame; `size` is normalized area.

## Listing COCO Class Names
```bash
python sports_ball_multitracker.py --list-classes
```

## Notes
- This tool depends on Ultralytics’ `ultralytics` package and its trackers.
