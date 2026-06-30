"""
gun / knife / bat detector — run on a video/image with object TRACKING.

Run from the repo root (inside conda env 'harmful'):
    conda run -n harmful python src/track.py video.mp4
    conda run -n harmful python src/track.py path/to/video.mp4 --show
    conda run -n harmful python src/track.py 0 --show              # webcam with tracking

Tracking assigns a stable ID to each detected object across frames (ByteTrack),
so the same gun/knife/bat keeps one ID instead of being re-counted every frame.
With --save, annotated output (boxes + IDs) goes to runs/detect/track*/.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "harmful_v3.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="video / image / folder path, or 0 for webcam")
    ap.add_argument("--conf", type=float, default=0.4,
                    help="confidence threshold (default 0.4; lower = more detections)")
    ap.add_argument("--device", default="mps", help="mps / cpu / 0 (cuda)")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="bytetrack.yaml (fast) or botsort.yaml (ReID, more robust)")
    ap.add_argument("--show", action="store_true", help="open a live annotated window")
    ap.add_argument("--save", action="store_true",
                    help="save the annotated output video (off by default)")
    args = ap.parse_args()

    source = 0 if args.source == "0" else args.source
    model = YOLO(MODEL_PATH)
    print(f"Model: {MODEL_PATH}  classes={model.names}  tracker={args.tracker}")

    # stream=True -> generator, frame-by-frame, constant memory.
    for r in model.track(source=source, conf=args.conf, device=args.device,
                         tracker=args.tracker, stream=True,
                         show=args.show, save=args.save, verbose=False):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        ids = r.boxes.id  # None until a track is confirmed
        for i, box in enumerate(r.boxes):
            cls = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            tid = int(ids[i]) if ids is not None else -1
            print(f"id={tid:>3}  {cls:<6} conf={conf:.2f}")


if __name__ == "__main__":
    main()
