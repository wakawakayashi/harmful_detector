# CLI Tools (`src/`)

Command-line scripts for the gun / knife / bat detector. They complement the
`app.py` web UI with terminal and webcam workflows, plus the dataset builders
used for (re)training.

All commands run **from the repo root** inside the conda env `harmful`
(prefix with `conda run -n harmful` or `conda activate harmful` first). Every
inference script loads the production model `models/harmful_v3.pt` automatically.

## Inference

| Script | Purpose | Example |
|--------|---------|---------|
| `predict.py` | One-shot detection on an image, video, folder, or webcam. Prints each detection (class, conf, box). | `python src/predict.py photo.jpg --conf 0.3` |
| `track.py` | Same, but with **ByteTrack** object IDs so each item is counted once across frames. | `python src/track.py video.mp4 --show` |
| `webcam.py` | **Live webcam** window with an on-screen FPS counter. Background thread keeps only the newest frame (no lag). | `python src/webcam.py --mirror` |

**Common flags**
- `--conf` — confidence threshold (default 0.4–0.45; lower = more detections).
- `--device` — `mps` (Mac GPU, default), `cpu`, or `0` (CUDA).
- `--save` — write annotated output to `runs/detect/...` (off by default).
- `--show` — open a live annotated window (`track.py`).
- `0` as the source — use the webcam (`predict.py`, `track.py`).

> Note: the CLI scripts use a single `--conf` for all classes. The Gradio app
> (`app.py`) additionally supports **per-class** thresholds (e.g. higher for `gun`,
> lower for `knife`) — handy for tuning the gun-vs-knife trade-off.

## Dataset builders (only needed to retrain)

| Script | Purpose |
|--------|---------|
| `build_dataset_v3.py` | **Current** — assembles the 3-class (gun/knife/bat) dataset by merging the curated + gun2 data with the Zenodo "Dangerous Items" set (remapped to 3 classes) and generates the YOLO data YAML (`configs/`, gitignored). |
| `build_dataset.py` | Legacy 2-class (gun/knife) builder that produced v2; kept for reference. |

These expect the raw datasets under `datasets/` (gitignored — supply your own).
The full data sources, class-mapping table, and dedup logic are documented in the
`build_dataset*.py` docstrings.
