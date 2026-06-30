# How It Works — Detection & Processing Pipeline

A technical walkthrough of how the app (`app.py`) turns an input image / video / YouTube
link into an annotated result: the per-class detection-and-filter strategy, color spaces,
coordinate handling, box drawing, object tracking, and browser-friendly video encoding.

Model: `models/harmful_v3.pt` (3-class — `0=gun`, `1=knife`, `2=bat`).

---

## 1. Setup & the core strategy (`app.py:22-29`)

```python
model = YOLO(MODEL_PATH)                              # harmful_v3.pt, 3-class
NAMES = model.names                                   # {0: gun, 1: knife, 2: bat}
COLORS = {0:(0,0,255), 1:(0,165,255), 2:(255,0,0)}    # BGR!
BASE_CONF = 0.20
```

Two key points:

- **Colors are BGR, not RGB** — all drawing uses OpenCV, which is BGR-native.
  `(0,0,255)` = red (gun), `(0,165,255)` = orange (knife), `(255,0,0)` = blue (bat).
- **`BASE_CONF = 0.20` is deliberately low.** Inference keeps every box `>= 0.20`, then a
  *second*, per-class filter is applied at draw time. This "detect low, filter per class"
  trick lets gun/knife/bat each have an independent threshold **without re-running
  inference** — impossible if a single `conf` were passed to the model. In practice:
  raise the `gun` threshold to cut its false positives, lower `knife` for recall.

---

## 2. `draw()` — per-class filter + annotation (`app.py:32-44`)

```python
c = int(box.cls[0]); conf = float(box.conf[0])
if conf < thr.get(c, 1.0): return None               # per-class gate
x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
cv2.putText(frame, label, (x1, max(12, y1-6)), ...)
```

- `box.xyxy[0]` is the box in **absolute pixel coordinates** (top-left `x1,y1` → bottom-right
  `x2,y2`), origin at the image's top-left. Floats are cast to `int` because OpenCV drawing
  needs integer pixels.
- `max(12, y1-6)` clamps the label's y so the text isn't drawn off the top edge when a box
  hugs the top of the frame.
- Returns the class id when a box survives the gate (so the caller can count it), or `None`
  when filtered out.

---

## 3. Image path (`run_image`, `app.py:79-92`)

```python
frame = cv2.imread(image_path)                        # BGR ndarray (H,W,3)
r = model.predict(frame, conf=BASE_CONF, ...)[0]
for b in r.boxes: draw(frame, b, thr)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

- `cv2.imread` decodes to a **BGR** NumPy array (or `None` if unreadable → raises).
- The critical final step is the **`BGR2RGB` conversion**: Gradio's `Image` component (like
  PIL) expects **RGB**. Skipping it would swap red and blue in every result. The video path
  does *not* do this conversion (see below) — the asymmetry is intentional.

---

## 4. Video path (`run_video`, `app.py:95-131`)

```python
cap = cv2.VideoCapture(src)        # read fps/w/h/frame-count, then release
writer = cv2.VideoWriter(raw_out, fourcc(*"mp4v"), fps, (w,h))
for i, r in enumerate(model.track(source=src, conf=BASE_CONF,
                                  tracker="bytetrack.yaml", stream=True, ...)):
    frame = r.orig_img.copy()
    ... draw boxes with track IDs ...
    writer.write(frame)            # BGR — no conversion
```

- `VideoCapture` is used **only to read metadata** (fps, width, height, total frames) — needed
  to build the writer and drive the progress bar — then released. Actual frame decoding happens
  inside `model.track`.
- **`stream=True`** makes `track()` a generator yielding one `Results` per frame, so a long
  video is never fully loaded into memory.
- `r.orig_img.copy()` — the original BGR frame; `.copy()` so drawing doesn't mutate
  ultralytics' internal buffer.
- `r.boxes.id` carries **ByteTrack IDs** (may be `None` before the tracker locks on), used to
  count *distinct objects* per class vs. just frames-with-detections.
- **No `BGR2RGB` here** — `cv2.VideoWriter` expects BGR input, and the output is consumed by
  Gradio as a *file*, not a raw array. Leaving it BGR is correct.

---

## 5. The H.264 re-encode (`to_h264`, `app.py:67-76`)

```bash
ffmpeg -i raw_out -c:v libx264 -pix_fmt yuv420p -movflags +faststart -an dst
```

`VideoWriter` writes **`mp4v`** (MPEG-4 Part 2), which browsers generally **won't** play in an
HTML5 `<video>` tag. So the raw file is transcoded once at the end:

- **`libx264`** → real H.264, which browsers play.
- **`-pix_fmt yuv420p`** → 4:2:0 chroma subsampling; many players reject the 4:4:4 OpenCV may emit.
- **`-movflags +faststart`** → moves the `moov` atom to the front so playback can start before
  the full download.
- **`-an`** → drops audio (only annotated visuals matter); falls back to the source on failure.

This is why there are **two temp files**: writing `mp4v` is fast and reliable via OpenCV's macOS
backend (direct H.264 from `VideoWriter` is unreliable there), so it writes fast, then transcodes
once at the end.

---

## Summary of the non-obvious bits

- Colors live in **BGR** throughout (OpenCV convention).
- The **image path converts to RGB** at the boundary; the **video path stays BGR** — different
  consumers (Gradio `Image` array vs. `VideoWriter`/file).
- Detection runs at a **low base conf (0.20)**; the *real* thresholds are applied **per class at
  draw time**, enabling independent gun/knife/bat tuning without re-inference.
- The result video is always **re-encoded to H.264 / yuv420p / faststart** purely so it plays in
  the browser.
