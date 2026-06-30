"""
gun / knife / bat detector — live webcam test.

Run from the repo root (inside conda env 'harmful'):
    conda run -n harmful python src/webcam.py
    conda run -n harmful python src/webcam.py --conf 0.4 --cam 0
    conda run -n harmful python src/webcam.py --mirror      # selfie (mirrored) view

A live annotated window opens. Press 'q' (with the window focused) to quit.
By default the view is NOT mirrored (real orientation, text reads correctly).
Add --mirror if you prefer the selfie-style flipped view.

The camera is read on a background thread that always keeps only the newest
frame (stale frames are dropped), so display/inference overhead never makes
the feed lag behind — same trick Ultralytics' own streaming loader uses.
"""
import argparse
import threading
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "harmful_v3.pt")


class LatestFrame:
    """Background camera reader that always serves the most recent frame."""

    def __init__(self, cam):
        self.cap = cv2.VideoCapture(cam)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open camera {cam} — check macOS camera permission.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                self.running = False
                break
            self.frame = f

    def read(self):
        return self.frame

    def release(self):
        self.running = False
        self.cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.4,
                    help="confidence threshold (default 0.4; lower = more detections)")
    ap.add_argument("--cam", type=int, default=0,
                    help="webcam index (default 0; try 1 for an external cam)")
    ap.add_argument("--device", default="mps", help="mps / cpu / 0 (cuda)")
    ap.add_argument("--mirror", action="store_true",
                    help="mirror the view horizontally (selfie style)")
    args = ap.parse_args()

    model = YOLO(MODEL_PATH)
    cam = LatestFrame(args.cam)
    print(f"Model: {MODEL_PATH}  classes={model.names}")
    print(f"mirror={args.mirror}  conf={args.conf}  device={args.device}")
    print("Live window opening... press 'q' in the window to quit.")

    while cam.read() is None:        # wait for the first frame
        time.sleep(0.01)

    win = "harmful detector (q to quit)"
    fps, last = 0.0, time.time()
    while True:
        frame = cam.read()
        if frame is None:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)  # horizontal flip

        r = model.predict(frame, conf=args.conf, device=args.device, verbose=False)[0]
        out = r.plot()

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        cv2.putText(out, f"{fps:4.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(win, out)

        if r.boxes is not None and len(r.boxes):
            print(" | ".join(f"{model.names[int(b.cls[0])]} {float(b.conf[0]):.2f}"
                              for b in r.boxes))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
