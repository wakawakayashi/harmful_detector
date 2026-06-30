"""
gun / knife / bat detector — inference script.

Run from the repo root (inside conda env 'harmful'):
    conda run -n harmful python src/predict.py photo.jpg
    conda run -n harmful python src/predict.py video.mp4 --conf 0.3
    conda run -n harmful python src/predict.py 0            # webcam (live window)
    conda run -n harmful python src/predict.py folder/      # batch

With --save, annotated outputs go to runs/detect/predict*/.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "harmful_v3.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="image / video / folder path, or 0 for webcam")
    ap.add_argument("--conf", type=float, default=0.45,
                    help="confidence threshold (default 0.45; try 0.30 for higher recall)")
    ap.add_argument("--device", default="mps", help="mps / cpu / 0 (cuda)")
    ap.add_argument("--save", action="store_true",
                    help="save annotated outputs to disk (off by default)")
    args = ap.parse_args()

    # turn the "0" string into int for webcam
    source = 0 if args.source == "0" else args.source

    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=source,
        conf=args.conf,
        device=args.device,
        save=args.save,
        show=(source == 0),   # open a live window for webcam
    )

    # summarize detections to the console
    names = model.names  # {0: 'gun', 1: 'knife', 2: 'bat'}
    for r in results:
        path = getattr(r, "path", "frame")
        if r.boxes is None or len(r.boxes) == 0:
            print(f"{path}: no detections")
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"{path}: {names[cls]}  conf={score:.2f}  "
                  f"box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")


if __name__ == "__main__":
    main()
