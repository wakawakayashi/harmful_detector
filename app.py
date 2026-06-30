"""
Harmful-item detector — drag-and-drop image / video / YouTube web UI (gun / knife / bat).

Launch:
    conda run -n harmful python app.py
Then open http://127.0.0.1:7860, drop an image OR a video OR paste a YouTube link, press Run.

Per-class confidence thresholds: raise 'gun' to cut its false positives, lower
'knife' for recall. Detection runs at a low base conf, then each box is kept only
if it clears its own class threshold. Video tracking (ByteTrack) gives every object
a stable ID; the output video is re-encoded to H.264 so it plays in the browser.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import gradio as gr
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = str(ROOT / "models" / "harmful_v3.pt")
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

model = YOLO(MODEL_PATH)
NAMES = model.names                       # {0: gun, 1: knife, 2: bat}
COLORS = {0: (0, 0, 255), 1: (0, 165, 255), 2: (255, 0, 0)}  # BGR: gun red, knife orange, bat blue
BASE_CONF = 0.20                          # detect low, then filter per class


def draw(frame, box, thr, ident=-1):
    """Draw a box if it clears its class threshold. Returns the class id or None."""
    c = int(box.cls[0])
    conf = float(box.conf[0])
    if conf < thr.get(c, 1.0):
        return None
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    color = COLORS.get(c, (0, 255, 0))
    label = f"{NAMES[c]} {conf:.2f}" + (f" #{ident}" if ident >= 0 else "")
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return c


def download_youtube(url: str, max_seconds: float) -> str:
    """Download a YouTube video (<=720p) to a temp file; optionally only the first N seconds."""
    d = tempfile.mkdtemp()
    cmd = ["yt-dlp", "-f", "best[height<=720][ext=mp4]/best[ext=mp4]/best",
           "-o", str(Path(d) / "yt.%(ext)s"), "--ffmpeg-location", FFMPEG,
           "--no-playlist", "--quiet", "--no-warnings"]
    if max_seconds and max_seconds > 0:
        cmd += ["--download-sections", f"*0-{int(max_seconds)}", "--force-keyframes-at-cuts"]
    cmd.append(url)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"YouTube download failed: {e.stderr[-300:] if e.stderr else e}")
    files = [p for p in Path(d).glob("yt.*")
             if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".m4v")]
    if not files:
        raise gr.Error("Could not download the YouTube video (link/access issue?).")
    return str(files[0])


def to_h264(src: str) -> str:
    """Re-encode to browser-playable H.264; fall back to the source on failure."""
    dst = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    try:
        subprocess.run([FFMPEG, "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", "-an", dst],
                       check=True, capture_output=True)
        return dst
    except Exception:
        return src


def run_image(image_path, thr, device):
    frame = cv2.imread(image_path)
    if frame is None:
        raise gr.Error("Could not read the image.")
    r = model.predict(frame, conf=BASE_CONF, device=device, verbose=False)[0]
    counts = {0: 0, 1: 0, 2: 0}
    if r.boxes is not None and len(r.boxes):
        for b in r.boxes:
            c = draw(frame, b, thr)
            if c is not None:
                counts[c] += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lines = [f"{NAMES[c]}: {counts[c]} box(es)" for c in (0, 1, 2) if counts[c]]
    return rgb, ("\n".join(lines) if lines else "No detections above thresholds.")


def run_video(src, thr, tracker, device, progress):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    raw_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    ids = {0: set(), 1: set(), 2: set()}
    frames_with = {0: 0, 1: 0, 2: 0}

    for i, r in enumerate(model.track(source=src, conf=BASE_CONF, device=device,
                                      tracker=tracker, stream=True, verbose=False)):
        frame = r.orig_img.copy()
        seen = set()
        if r.boxes is not None and len(r.boxes):
            tid = r.boxes.id
            for j, b in enumerate(r.boxes):
                ident = int(tid[j]) if tid is not None else -1
                c = draw(frame, b, thr, ident)
                if c is not None:
                    seen.add(c)
                    if ident >= 0:
                        ids[c].add(ident)
        for c in seen:
            frames_with[c] += 1
        writer.write(frame)
        if total:
            progress((i + 1) / total, desc="Processing")

    writer.release()
    playable = to_h264(raw_out)
    lines = [f"{NAMES[c]}: {frames_with[c]} frames, {len(ids[c])} distinct object(s)"
             for c in (0, 1, 2) if frames_with[c]]
    return playable, ("\n".join(lines) if lines else "No detections above thresholds.")


def process(image_path, video_path, youtube_url, max_seconds,
            gun_t, knife_t, bat_t, tracker, device, progress=gr.Progress()):
    thr = {0: gun_t, 1: knife_t, 2: bat_t}

    if image_path:
        img, summary = run_image(image_path, thr, device)
        return img, None, summary

    if youtube_url and youtube_url.strip():
        progress(0, desc="Downloading from YouTube")
        src = download_youtube(youtube_url.strip(), max_seconds)
    elif video_path:
        src = video_path
    else:
        return None, None, "Drop an image/video or paste a YouTube link."

    video, summary = run_video(src, thr, tracker, device, progress)
    return None, video, summary


with gr.Blocks(title="Harmful-item detector") as demo:
    gr.Markdown(
        "## Harmful-item detector — gun · knife · bat\n"
        "Drag & drop an **image** or **video**, or paste a **YouTube link**; "
        "adjust the thresholds, then **Run**."
    )
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="Image (drag & drop)", type="filepath")
            vid_in = gr.Video(label="...or a video (drag & drop)")
            yt_url = gr.Textbox(label="...or a YouTube link",
                                placeholder="https://www.youtube.com/watch?v=...")
            max_sec = gr.Number(value=60, label="YouTube: first N seconds (0 = whole video)", precision=0)
            gun_t = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="gun threshold")
            knife_t = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="knife threshold")
            bat_t = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="bat threshold")
            tracker = gr.Dropdown(["bytetrack.yaml", "botsort.yaml"],
                                  value="bytetrack.yaml", label="tracker (video)")
            device = gr.Dropdown(["mps", "cpu"], value="mps", label="device")
            run = gr.Button("Run", variant="primary")
        with gr.Column():
            img_out = gr.Image(label="Result (image)")
            vid_out = gr.Video(label="Result (video — boxes + tracking IDs)")
            summary = gr.Textbox(label="Summary", lines=4)

    run.click(process,
              [img_in, vid_in, yt_url, max_sec, gun_t, knife_t, bat_t, tracker, device],
              [img_out, vid_out, summary])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
