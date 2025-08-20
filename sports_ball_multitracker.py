#!/usr/bin/env python3
"""
YOLO Multi-Object Tracker → OSC (x, y, size per slot)

- Uses Ultralytics YOLO (v8/v11/etc.) + BYTETracker/BoT-SORT to track up to N objects
  from COCO classes (default: "sports ball") with persistent IDs across frames.
- Emits per-slot OSC messages to control external software (e.g., Max/MSP):
    /ball/1 [x_norm, y_norm, size_norm]
    /ball/2 [x_norm, y_norm, size_norm]
    /ball/3 [x_norm, y_norm, size_norm]
  where:
    x_norm, y_norm ∈ [0..1] with origin at top-left of the video frame
    size_norm = bbox_area / frame_area (proxy for distance/scale)
- A slot is a stable integer tag (1..max-slots). The script maps YOLO's tracker IDs
  to slots and tries to keep them stable; new/larger detections can replace the
  smallest-current slot when all slots are full.

Key features
------------
- Multi-class support via --classes "sports ball,cup,banana"
- Fast real-time performance on CPU; GPU/CUDA/MPS optional
- EMA smoothing per slot for stability (--ema)
- Short-term hold on brief dropouts (--hold)
- Minimum-size gate to reject tiny/weak boxes (--min-area)
- Class listing helper (--list-classes)

Typical usage
-------------
# Track tennis balls (up to 3), send OSC to localhost:9000
python tennis_ball_controller_multitrack.py --classes "sports ball" --max-slots 3 --imgsz 320 --ema 0.25

# Track tennis balls OR cups OR bananas (whichever appear), up to 3 at once
python tennis_ball_controller_multitrack.py --classes "sports ball,cup,banana" --max-slots 3

# Print available class names and exit
python tennis_ball_controller_multitrack.py --list-classes
"""

import cv2
import time
import argparse
from ultralytics import YOLO
from pythonosc.udp_client import SimpleUDPClient

try:
    import torch  # Optional (for device detection; works without)
except Exception:
    torch = None


def parse_args():
    """Define and parse command-line arguments."""
    ap = argparse.ArgumentParser(description="Track up to N objects with YOLO+Tracker → OSC.")
    # Model & source
    ap.add_argument("--model", default="yolov8n.pt",
                    help="Ultralytics YOLO model (COCO). e.g., yolov8n.pt, yolov8s.pt, yolo11n.pt, yolo11s.pt")
    ap.add_argument("--source", default="0",
                    help="Camera index (0/1/...) or video path/URL.")
    ap.add_argument("--device", default="auto",
                    help="Inference device: cpu | cuda | 0 | 1 | mps | auto")
    ap.add_argument("--imgsz", type=int, default=320,
                    help="Inference image size (short side). 256–384 are good CPU values.")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Minimum detection confidence.")
    ap.add_argument("--iou", type=float, default=0.45,
                    help="NMS IoU threshold.")
    ap.add_argument("--half", action="store_true",
                    help="Half precision (CUDA only). Ignored on CPU/MPS.")

    # Smoothing / stability
    ap.add_argument("--ema", type=float, default=0.25,
                    help="EMA smoothing factor per slot (0..1). 0 disables.")
    ap.add_argument("--hold", type=int, default=12,
                    help="Frames to hold last value on brief misses before sending sentinel.")
    ap.add_argument("--min-area", type=float, default=0.0008,
                    help="Minimum normalized bbox area to accept (e.g., 0.0008 = 0.08% of frame).")

    # Runtime / display
    ap.add_argument("--fps-cap", type=float, default=0.0,
                    help="Cap processing FPS (0 = uncapped).")
    ap.add_argument("--no-video", action="store_true",
                    help="Disable preview window (headless).")
    ap.add_argument("--overlay", action="store_true",
                    help="Force overlay drawing even with --no-video (useful for file output).")

    # OSC
    ap.add_argument("--osc-host", default="127.0.0.1",
                    help="OSC destination host.")
    ap.add_argument("--osc-port", type=int, default=9000,
                    help="OSC destination port.")
    ap.add_argument("--base-path", default="/ball",
                    help="Base OSC path for per-slot data (e.g., /ball/1).")
    ap.add_argument("--count-path", default="/balls/count",
                    help="OSC path that reports number of active (non-sentinel) slots.")

    # Slots / tracking
    ap.add_argument("--max-slots", type=int, default=3,
                    help="Maximum concurrent objects to output (slots 1..N).")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="Ultralytics tracker config (e.g., bytetrack.yaml, botsort.yaml).")

    # Classes
    ap.add_argument("--classes", default="sports ball",
                    help='Comma-separated COCO class names to track, e.g. "sports ball,cup,banana".')
    ap.add_argument("--list-classes", action="store_true",
                    help="Print available COCO class names and exit.")
    return ap.parse_args()


def choose_device(arg: str) -> str:
    """Resolve device selection ('auto' → cuda if available, else cpu)."""
    if arg != "auto":
        return arg
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    # On Apple Silicon you can pass --device mps explicitly if desired
    return "cpu"


def open_capture(source: str) -> cv2.VideoCapture:
    """Open a camera index or video file/URL with sane defaults for webcam."""
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
        # Modest capture size helps CPU performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    except ValueError:
        cap = cv2.VideoCapture(source)
    return cap


class SlotState:
    """Per-slot smoothing/hold state."""
    def __init__(self):
        self.ema = [None, None, None]       # [x, y, size]
        self.last_out = [-1.0, -1.0, 0.0]   # sentinel until first valid value
        self.misses = 0
        self.has_value = False              # whether we have emitted a valid value


def main():
    args = parse_args()

    # Load model and set device/precision
    model = YOLO(args.model)
    device = choose_device(args.device)
    model.to(device)
    if args.half and device == "cuda":
        # FP16 is CUDA-only in this context
        try:
            model.model.half()
        except Exception:
            pass

    # Class name dictionary from model
    names = getattr(model, "names", None) or getattr(model.model, "names", None)
    if not isinstance(names, dict):
        raise RuntimeError("Could not access model class names.")
    id2name = {int(i): str(n) for i, n in names.items()}
    name2id = {str(n).lower(): int(i) for i, n in names.items()}

    # Optional: list classes and exit
    if args.list_classes:
        print("Available classes:")
        for i in sorted(id2name):
            print(f"{i:2d}: {id2name[i]}")
        return

    # Map requested class names → IDs
    wanted = [c.strip().lower() for c in args.classes.split(",") if c.strip()]
    target_ids = [name2id[c] for c in wanted if c in name2id]
    if not target_ids:
        sample = ", ".join(list(name2id.keys())[:10])
        raise RuntimeError(f"No matching classes for --classes '{args.classes}'. "
                           f"Examples: {sample} ...")

    # OSC client
    client = SimpleUDPClient(args.osc_host, args.osc_port)

    # Video source
    cap = open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    # Track-ID ↔ slot mapping and state
    track_to_slot = {}                        # tracker ID -> slot index
    slot_to_track = {}                        # slot index -> tracker ID
    slots = {i: SlotState() for i in range(1, args.max_slots + 1)}

    # FPS throttling / display
    frame_interval = (1.0 / args.fps_cap) if args.fps_cap > 0 else 0.0
    last_tick = 0.0
    fps_show, fps_t0, fps_count = 0.0, time.time(), 0

    # UI window
    title = "YOLO Multi-Track (OSC)"
    if not args.no_video:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    try:
        while True:
            # Optional FPS cap to reduce CPU load and stabilize timing
            if frame_interval > 0:
                now = time.time()
                wait = frame_interval - (now - last_tick)
                if wait > 0:
                    time.sleep(wait)
                last_tick = time.time()

            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]

            # ---- Tracking inference (persistent IDs) ----
            results = model.track(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=target_ids,        # filter to the class IDs requested
                tracker=args.tracker,      # e.g., bytetrack.yaml or botsort.yaml
                persist=True,              # keep tracker state across frames
                verbose=False,
                max_det=max(3, args.max_slots),
                agnostic_nms=False
            )

            # Collect current detections with tracker IDs
            dets = []  # each: {track_id, cx, cy, area, box, cls}
            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    ids = None
                    if hasattr(r.boxes, "id") and r.boxes.id is not None:
                        ids = r.boxes.id.cpu().numpy().astype(int)
                    for i, b in enumerate(r.boxes):
                        # Confidence guard (trackers can pass a few low-conf)
                        conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
                        if conf < args.conf:
                            continue
                        # Class guard (should already be filtered)
                        cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                        if cls_id not in target_ids:
                            continue

                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                        bw = max(0.0, x2 - x1)
                        bh = max(0.0, y2 - y1)
                        area = bw * bh
                        cx = (x1 + x2) * 0.5
                        cy = (y1 + y2) * 0.5

                        # Track ID (can be -1 when unassigned)
                        track_id = None
                        if ids is not None and i < len(ids):
                            tid = int(ids[i])
                            if tid >= 0:
                                track_id = tid

                        if track_id is not None:
                            dets.append({
                                "track_id": track_id,
                                "cx": cx, "cy": cy,
                                "area": area,
                                "box": (int(x1), int(y1), int(x2), int(y2)),
                                "cls": cls_id
                            })

            # Keep at most --max-slots detections (largest areas = closest/most visible)
            if len(dets) > args.max_slots:
                dets.sort(key=lambda d: d["area"], reverse=True)
                dets = dets[:args.max_slots]

            # Slot selection logic:
            # 1) Keep existing track->slot assignments where possible
            # 2) Fill free slots with new tracks
            # 3) If no free slots, replace the smallest-area slot if the new detection is larger
            slot_area = {s: 0.0 for s in range(1, args.max_slots + 1)}
            for d in dets:
                if d["track_id"] in track_to_slot:
                    s = track_to_slot[d["track_id"]]
                    slot_area[s] = d["area"]

            for d in dets:
                tid = d["track_id"]
                if tid in track_to_slot:
                    continue
                free_slot = next((s for s in range(1, args.max_slots + 1) if s not in slot_to_track), None)
                if free_slot is not None:
                    track_to_slot[tid] = free_slot
                    slot_to_track[free_slot] = tid
                    slot_area[free_slot] = d["area"]
                else:
                    smallest_slot = min(slot_area, key=lambda s: slot_area[s])
                    if d["area"] > slot_area[smallest_slot]:
                        old_tid = slot_to_track.get(smallest_slot)
                        if old_tid is not None:
                            track_to_slot.pop(old_tid, None)
                        track_to_slot[tid] = smallest_slot
                        slot_to_track[smallest_slot] = tid
                        slot_area[smallest_slot] = d["area"]

            # Build slot->detection map for this frame
            slot_det = {s: None for s in range(1, args.max_slots + 1)}
            for d in dets:
                s = track_to_slot.get(d["track_id"], None)
                if s is not None and 1 <= s <= args.max_slots:
                    slot_det[s] = d

            # Emit per-slot data (with EMA, hold, and sentinels)
            n_active = 0
            drew_overlay = False
            for s in range(1, args.max_slots + 1):
                st = slots[s]
                d = slot_det[s]
                path = f"{args.base_path}/{s}"

                if d is not None:
                    x_norm = float(d["cx"]) / float(w)
                    y_norm = float(d["cy"]) / float(h)
                    size_norm = float(d["area"]) / float(w * h)

                    if size_norm >= args.min_area:
                        st.misses = 0
                        if args.ema > 0:
                            a = args.ema
                            if st.ema[0] is None:
                                st.ema = [x_norm, y_norm, size_norm]
                            else:
                                st.ema[0] = (1 - a) * st.ema[0] + a * x_norm
                                st.ema[1] = (1 - a) * st.ema[1] + a * y_norm
                                st.ema[2] = (1 - a) * st.ema[2] + a * size_norm
                            out = [st.ema[0], st.ema[1], st.ema[2]]
                        else:
                            out = [x_norm, y_norm, size_norm]

                        st.last_out = out
                        st.has_value = True
                        client.send_message(path, out)
                        n_active += 1

                        # Optional overlay
                        if (not args.no_video) or args.overlay:
                            x1, y1, x2, y2 = d["box"]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.circle(frame, (int(d["cx"]), int(d["cy"])), 6, (0, 255, 0), -1)
                            label = f"{s}:{id2name.get(d['cls'],'?')}"
                            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            drew_overlay = True
                    else:
                        d = None  # fall through to "miss" branch below

                if d is None:
                    # No fresh detection for this slot this frame
                    if st.has_value and st.misses < args.hold:
                        st.misses += 1
                        client.send_message(path, st.last_out)  # hold last known value
                        if st.last_out[2] > 0:
                            n_active += 1
                    else:
                        # Send sentinel and reset smoothing
                        st.last_out = [-1.0, -1.0, 0.0]
                        st.ema = [None, None, None]
                        st.has_value = False
                        client.send_message(path, st.last_out)

            # Report number of active (non-sentinel) slots
            client.send_message(args.count_path, n_active)

            # FPS overlay
            fps_count += 1
            if fps_count >= 10:
                t1 = time.time()
                fps_show = fps_count / (t1 - fps_t0)
                fps_t0 = t1
                fps_count = 0

            # Preview window
            if not args.no_video:
                txt = f"FPS {fps_show:.1f}   conf>={args.conf}  imgsz={args.imgsz}  active={n_active}"
                color = (0, 255, 0) if drew_overlay else (0, 0, 255)
                cv2.putText(frame, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if not drew_overlay:
                    cv2.putText(frame, "No targets (holding prior slots where possible)",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                cv2.imshow(title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # Esc or q
                    break

    finally:
        # Clean up resources
        cap.release()
        if not args.no_video:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
