"""
Batch-test a FOLDER of videos with TRACKING.

Drop your videos into one folder, then run from the repo root:
    conda run -n harmful python src/batch_videos.py path/to/folder
    conda run -n harmful python src/batch_videos.py path/to/folder --conf 0.3
    conda run -n harmful python src/batch_videos.py path/to/folder --show   # live window per video

Runs the model on EVERY video in the folder (one after another), with ByteTrack
IDs, and prints a per-video summary: frames, detection frames, per-class frame
counts, peak confidence, and how many DISTINCT objects were tracked (via track
IDs, so the same item isn't re-counted). With --save, each annotated result is
written under runs/detect/track*/.

Points to the v3 model (classes: gun, knife, bat) in ../models/harmful_v3.pt.
"""
import argparse
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "harmful_v3.pt")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder containing video files")
    ap.add_argument("--conf", type=float, default=0.4,
                    help="confidence threshold (default 0.4; lower = more detections)")
    ap.add_argument("--device", default="mps", help="mps / cpu / 0 (cuda)")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="bytetrack.yaml (fast) or botsort.yaml (ReID)")
    ap.add_argument("--show", action="store_true", help="open a live window per video")
    ap.add_argument("--save", action="store_true",
                    help="save annotated output videos (off by default)")
    args = ap.parse_args()

    folder = Path(args.folder)
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise SystemExit(f"No videos in {folder} (looked for {sorted(VIDEO_EXTS)})")

    model = YOLO(MODEL_PATH)
    names = model.names
    print(f"Model: {MODEL_PATH}  classes={names}")
    print(f"Found {len(videos)} video(s) in {folder}\n")

    report = []
    for vi, video in enumerate(videos, 1):
        print(f"[{vi}/{len(videos)}] tracking {video.name} ...")
        frames = det_frames = 0
        cls_frames = defaultdict(int)   # frames containing each class
        peak = defaultdict(float)       # peak confidence per class
        ids = defaultdict(set)          # distinct track ids per class

        for r in model.track(source=str(video), conf=args.conf, device=args.device,
                             tracker=args.tracker, stream=True,
                             show=args.show, save=args.save, verbose=False):
            frames += 1
            if r.boxes is None or len(r.boxes) == 0:
                continue
            det_frames += 1
            tid = r.boxes.id
            seen = set()
            for i, box in enumerate(r.boxes):
                c = int(box.cls[0])
                seen.add(c)
                peak[c] = max(peak[c], float(box.conf[0]))
                if tid is not None:
                    ids[c].add(int(tid[i]))
            for c in seen:
                cls_frames[c] += 1

        report.append((video.name, frames, det_frames,
                       dict(cls_frames), dict(peak),
                       {c: len(s) for c, s in ids.items()}))

    print("\n==================== SUMMARY ====================")
    for name, frames, det_frames, cls_frames, peak, idcount in report:
        print(f"\n{name}  ({frames} frames, {det_frames} with detections)")
        if not cls_frames:
            print("   no detections")
            continue
        for c in sorted(cls_frames):
            print(f"   {names[c]:6}: {cls_frames[c]:5} frames | peak {peak[c]:.2f} | "
                  f"{idcount.get(c, 0)} distinct tracked")


if __name__ == "__main__":
    main()
